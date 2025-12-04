import numpy as np
from modules import (
    Parameter, Linear, LayerNorm, GELU, SiLU,
    Dropout, MultiHeadAttention, 
    PositionalEncoding, LearnablePositionalEncoding, ConditionalPositionalEncoding
)


class MLP:
    """Standard Feed-forward network (MLP) with GELU"""

    def __init__(self, d_model: int, mlp_ratio: float = None, dropout: float = 0.1):
        dim_hidden = int(d_model * mlp_ratio)
        self.fc1 = Linear(d_model, dim_hidden)
        self.gelu = GELU()
        self.dropout1 = Dropout(dropout)
        self.fc2 = Linear(dim_hidden, d_model)
        self.dropout2 = Dropout(dropout)

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = self.fc1.forward(x)
        x = self.gelu.forward(x)
        x = self.dropout1.forward(x)
        x = self.fc2.forward(x)
        x = self.dropout2.forward(x)
        return x

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        grad = self.dropout2.backward(grad_output)
        grad = self.fc2.backward(grad)
        grad = self.dropout1.backward(grad)
        grad = self.gelu.backward(grad)
        grad = self.fc1.backward(grad)
        return grad

    def parameters(self):
        return self.fc1.parameters() + self.fc2.parameters()

    def train(self):
        self.dropout1.train()
        self.dropout2.train()

    def eval(self):
        self.dropout1.eval()
        self.dropout2.eval()


class SwiGLU:
    """
    SwiGLU: Gated Linear Unit with SiLU activation

    FFN_SwiGLU(x) = (SiLU(xW_gate) ⊙ xW_value) W_out

    Typically uses hidden_dim = 8/3 * d_model (≈ 2.67x)
    """

    def __init__(self, d_model: int, mlp_ratio: float = None, dropout: float = 0.1):
        if mlp_ratio is None:
            # Default: 8/3 * d_model for SwiGLU
            hidden_dim = int(8 * d_model / 3)
            # Round to nearest multiple of 8 for efficiency
            hidden_dim = ((hidden_dim + 7) // 8) * 8
        else:
            hidden_dim = int(d_model * mlp_ratio)

        self.d_model = d_model
        self.hidden_dim = hidden_dim

        # Gate and value projections
        self.gate_proj = Linear(d_model, hidden_dim)
        self.value_proj = Linear(d_model, hidden_dim)
        self.out_proj = Linear(hidden_dim, d_model)

        self.silu = SiLU()
        self.dropout = Dropout(dropout)

        # Cache for backward
        self.cache = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        gate = self.gate_proj.forward(x)
        gate = self.silu.forward(gate)

        value = self.value_proj.forward(x)

        hidden = gate * value

        output = self.out_proj.forward(hidden)
        output = self.dropout.forward(output)

        self.cache = {'gate': gate, 'value': value}
        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        grad_output = self.dropout.backward(grad_output)

        grad_hidden = self.out_proj.backward(grad_output)

        gate = self.cache['gate']
        value = self.cache['value']

        grad_gate = grad_hidden * value
        grad_value = grad_hidden * gate

        grad_x_value = self.value_proj.backward(grad_value)

        grad_gate = self.silu.backward(grad_gate)
        grad_x_gate = self.gate_proj.backward(grad_gate)

        grad_x = grad_x_gate + grad_x_value
        return grad_x

    def parameters(self):
        return (self.gate_proj.parameters() +
                self.value_proj.parameters() +
                self.out_proj.parameters())

    def train(self):
        self.dropout.train()

    def eval(self):
        self.dropout.eval()

class TransformerEncoderLayer:
    """Single Transformer encoder layer with manual backprop"""

    def __init__(self, d_model: int, nhead: int, mlp_ratio: float = None, dropout: float = 0.1):
        # Multi-head attention
        self.norm1 = LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, nhead, dropout)

        # Feed-forward
        self.norm2 = LayerNorm(d_model)
        # self.ffn = MLP(d_model, mlp_ratio, dropout)
        self.ffn = SwiGLU(d_model, mlp_ratio, dropout)

    def forward(self, x: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: (batch, seq_len) boolean mask where True indicates valid positions
        Returns:
            output: (batch, seq_len, d_model)
        """
        # Self-attention with residual
        attn_input = self.norm1.forward(x)
        attn_output = self.attn.forward(attn_input, mask=mask)
        x = x + attn_output  # Residual connection

        # Feed-forward with residual
        ff_input = self.norm2.forward(x)
        ff_output = self.ffn.forward(ff_input)
        x = x + ff_output  # Residual connection

        return x

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Args:
            grad_output: (batch, seq_len, d_model)
        Returns:
            grad_input: (batch, seq_len, d_model)
        """
        # Backward through second residual connection
        grad_ff_output = grad_output
        grad_x2 = grad_output

        # Backward through feed-forward
        grad_ff = self.ffn.backward(grad_ff_output)
        grad_ff_input = self.norm2.backward(grad_ff)
        grad_x2 = grad_x2 + grad_ff_input

        # Backward through first residual connection
        grad_attn_output = grad_x2
        grad_x1 = grad_x2

        # Backward through attention
        grad_attn = self.attn.backward(grad_attn_output)
        grad_attn_input = self.norm1.backward(grad_attn)
        grad_x1 = grad_x1 + grad_attn_input

        return grad_x1

    def parameters(self):
        return (self.norm1.parameters() + self.attn.parameters() +
                self.norm2.parameters() + self.ffn.parameters())

    def train(self):
        self.attn.train()
        self.ffn.train()
    def eval(self):
        self.attn.eval()
        self.ffn.eval()


class Transformer:
    """
    Transformer for classification with manual backpropagation

    Now includes:
    - Positional encoding (crucial for sequence understanding)
    - CLS token (learnable global representation)
    """

    def __init__(self, input_dim: int = 3, d_model: int = 128, nhead: int = 4,
                 num_layers: int = 2, mlp_ratio: float = None,
                 dropout: float = 0.1, num_classes: int = 10,
                 pos_encoding: str = 'sinusoidal'):

        self.input_dim = input_dim
        self.d_model = d_model
        self.num_classes = num_classes

        # Input projection
        self.input_proj = Linear(input_dim, d_model)

        # CLS token (learnable parameter)
        self.cls_token = Parameter(np.random.randn(1, 1, d_model).astype(np.float32) * 0.02)

        # Positional encoding (max_len = 513 to account for CLS token)
        
        max_len = 513  # Account for CLS token
        if pos_encoding == 'sinusoidal':
            self.pos_encoder = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)
        elif pos_encoding == 'learnable':
            self.pos_encoder = LearnablePositionalEncoding(d_model, max_len=max_len, dropout=dropout)
        elif pos_encoding == 'conditional':
            self.pos_encoder = ConditionalPositionalEncoding(d_model, kernel_size=3, dropout=dropout)
        else:
            raise ValueError(f"Unknown pos_encoding: {pos_encoding}")

        # Transformer layers
        self.layers = [
            TransformerEncoderLayer(d_model, nhead, mlp_ratio, dropout)
            for _ in range(num_layers)
        ]

        # Output layers
        self.norm = LayerNorm(d_model)
        self.head = Linear(d_model, num_classes)

        # Cache for backward
        self.cache = {}

    def forward(self, x: np.ndarray, mask: np.ndarray = None, training: bool = True) -> np.ndarray:
        """
        Forward pass

        Args:
            x: (batch, seq_len, input_dim)
            mask: (batch, seq_len) boolean mask where True indicates valid positions
            training: whether in training mode (affects dropout)
        Returns:
            logits: (batch, num_classes)
        """
        batch_size = x.shape[0]

        # Set training mode
        if training:
            self.train()
        else:
            self.eval()

        # Project input
        x = self.input_proj.forward(x)  # (batch, seq_len, d_model)

        # Prepend CLS token
        cls_tokens = np.tile(self.cls_token.data, (batch_size, 1, 1))  # (batch, 1, d_model)
        x = np.concatenate([cls_tokens, x], axis=1)  # (batch, seq_len+1, d_model)

        # Update mask to account for CLS token (CLS is always valid)
        if mask is not None:
            cls_mask = np.ones((batch_size, 1), dtype=mask.dtype)
            mask = np.concatenate([cls_mask, mask], axis=1)  # (batch, seq_len+1)

        # Add positional encoding
        x = self.pos_encoder.forward(x)  # (batch, seq_len+1, d_model)
        self.cache['after_pos'] = x

        # Transformer layers
        for layer in self.layers:
            x = layer.forward(x, mask=mask)

        # Extract CLS token output
        cls_out = x[:, 0, :]  # (batch, d_model)

        # Normalization and classification head
        cls_out = self.norm.forward(cls_out)
        logits = self.head.forward(cls_out)

        return logits

    def backward(self, grad_logits: np.ndarray) -> np.ndarray:
        """
        Backward pass

        Args:
            grad_logits: (batch, num_classes)
        Returns:
            grad_input: (batch, seq_len, input_dim)
        """
        # Backward through classification head
        grad_cls_out = self.head.backward(grad_logits)  # (batch, d_model)
        grad_cls_out = self.norm.backward(grad_cls_out)

        # Backward through CLS token extraction
        # grad_cls_out goes to position 0, rest get zeros
        batch_size, seq_len_plus_1, d_model = self.cache['after_pos'].shape
        grad_x = np.zeros((batch_size, seq_len_plus_1, d_model), dtype=np.float32)
        grad_x[:, 0, :] = grad_cls_out  # Only CLS position gets gradient

        # Backward through transformer layers (reverse order)
        for layer in reversed(self.layers):
            grad_x = layer.backward(grad_x)

        # Backward through positional encoding (gradient passes through)
        grad_x = self.pos_encoder.backward(grad_x)

        # Split gradient: CLS token and input
        grad_cls_token = grad_x[:, 0:1, :]  # (batch, 1, d_model)
        grad_input_proj = grad_x[:, 1:, :]  # (batch, seq_len, d_model)

        # Accumulate gradient for CLS token (sum over batch)
        self.cls_token.grad += grad_cls_token.sum(axis=0)  # (1, d_model)

        # Backward through input projection
        grad_input = self.input_proj.backward(grad_input_proj)

        return grad_input

    def parameters(self):
        """Return all parameters (including CLS token)"""
        params = [self.cls_token] + self.input_proj.parameters()
        for layer in self.layers:
            params.extend(layer.parameters())
        params.extend(self.norm.parameters())
        params.extend(self.head.parameters())
        return params

    def train(self):
        """Set to training mode"""
        self.pos_encoder.train()
        for layer in self.layers:
            layer.train()

    def eval(self):
        """Set to evaluation mode"""
        self.pos_encoder.eval()
        for layer in self.layers:
            layer.eval()
