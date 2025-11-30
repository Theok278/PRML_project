"""
Transformer Model with Manual Backpropagation

Contains FeedForward, TransformerEncoderLayer, and SimplifiedTransformer.
"""

import numpy as np
from modules import (
    Parameter, Linear, LayerNorm, GELU,
    Dropout, MultiHeadAttention, PositionalEncoding
)


class FeedForward:
    """Feed-forward network (MLP) with manual backprop"""

    def __init__(self, d_model: int, dim_feedforward: int, dropout: float = 0.1):
        self.fc1 = Linear(d_model, dim_feedforward)
        self.gelu = GELU()
        self.dropout1 = Dropout(dropout)
        self.fc2 = Linear(dim_feedforward, d_model)
        self.dropout2 = Dropout(dropout)  # Add second dropout

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = self.fc1.forward(x)
        x = self.gelu.forward(x)
        x = self.dropout1.forward(x)
        x = self.fc2.forward(x)
        x = self.dropout2.forward(x)  # Add dropout after fc2
        return x

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        grad = self.dropout2.backward(grad_output)  # Backward through dropout2
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


class TransformerEncoderLayer:
    """Single Transformer encoder layer with manual backprop"""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        # Multi-head attention
        self.norm1 = LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, nhead, dropout)

        # Feed-forward
        self.norm2 = LayerNorm(d_model)
        self.ff = FeedForward(d_model, dim_feedforward, dropout)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            output: (batch, seq_len, d_model)
        """
        # Self-attention with residual
        attn_input = self.norm1.forward(x)
        attn_output = self.attn.forward(attn_input)
        x = x + attn_output  # Residual connection

        # Feed-forward with residual
        ff_input = self.norm2.forward(x)
        ff_output = self.ff.forward(ff_input)
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
        grad_ff = self.ff.backward(grad_ff_output)
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
                self.norm2.parameters() + self.ff.parameters())

    def train(self):
        self.attn.train()
        self.ff.train()

    def eval(self):
        self.attn.eval()
        self.ff.eval()


class Transformer:
    """
    Transformer for classification with manual backpropagation

    Now includes:
    - Positional encoding (crucial for sequence understanding)
    - CLS token (learnable global representation)
    """

    def __init__(self, input_dim: int = 3, d_model: int = 128, nhead: int = 4,
                 num_layers: int = 2, dim_feedforward: int = 256,
                 dropout: float = 0.1, num_classes: int = 10):

        self.input_dim = input_dim
        self.d_model = d_model
        self.num_classes = num_classes

        # Input projection
        self.input_proj = Linear(input_dim, d_model)

        # CLS token (learnable parameter)
        self.cls_token = Parameter(np.random.randn(1, 1, d_model).astype(np.float32) * 0.02)

        # Positional encoding (max_len = 513 to account for CLS token)
        self.pos_encoder = PositionalEncoding(d_model, max_len=513)

        # Transformer layers
        self.layers = [
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ]

        # Output layers
        self.norm = LayerNorm(d_model)
        self.head = Linear(d_model, num_classes)

        # Cache for backward
        self.cache = {}

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Forward pass

        Args:
            x: (batch, seq_len, input_dim)
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

        # Add positional encoding
        x = self.pos_encoder.forward(x)  # (batch, seq_len+1, d_model)
        self.cache['after_pos'] = x

        # Transformer layers
        for layer in self.layers:
            x = layer.forward(x)

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
        for layer in self.layers:
            layer.train()

    def eval(self):
        """Set to evaluation mode"""
        for layer in self.layers:
            layer.eval()
