"""
Transformer model with SupCon support

This model can output both embeddings (for contrastive learning)
and logits (for classification).
"""

import numpy as np
from modules import (
    Parameter, Linear, LayerNorm, GELU, SiLU,
    Dropout, MultiHeadAttention,
    PositionalEncoding, LearnablePositionalEncoding, ConditionalPositionalEncoding
)
from model import MLP, SwiGLU, TransformerEncoderLayer


class ProjectionHead:
    """
    Projection head for contrastive learning

    Maps encoder output to a lower-dimensional space where
    contrastive loss is computed.

    Architecture: Linear -> ReLU -> Linear
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        """
        Args:
            input_dim: Dimension of encoder output
            hidden_dim: Hidden dimension
            output_dim: Output embedding dimension
        """
        self.fc1 = Linear(input_dim, hidden_dim)
        self.fc2 = Linear(hidden_dim, output_dim)
        self.cache = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass

        Args:
            x: (batch, input_dim)
        Returns:
            embeddings: (batch, output_dim)
        """
        h = self.fc1.forward(x)
        h = np.maximum(0, h)  # ReLU
        self.cache['h'] = h
        self.cache['x'] = x
        embeddings = self.fc2.forward(h)
        return embeddings

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass

        Args:
            grad_output: (batch, output_dim)
        Returns:
            grad_input: (batch, input_dim)
        """
        grad_h = self.fc2.backward(grad_output)

        # ReLU backward
        h = self.cache['h']
        grad_h = grad_h * (h > 0)

        grad_x = self.fc1.backward(grad_h)
        return grad_x

    def parameters(self):
        return self.fc1.parameters() + self.fc2.parameters()

    def train(self):
        pass

    def eval(self):
        pass


class TransformerSupCon:
    """
    Transformer for Supervised Contrastive Learning

    Architecture:
        1. Encoder: Transformer encoder (shared)
        2. Projection Head: For contrastive learning
        3. Classification Head: For supervised classification

    Training can be done in two modes:
        - Joint: Train both losses simultaneously
        - Two-stage: First train with SupCon, then fine-tune classifier
    """

    def __init__(self, input_dim: int = 3, d_model: int = 128, nhead: int = 4,
                 num_layers: int = 2, mlp_ratio: float = None,
                 dropout: float = 0.1, num_classes: int = 10,
                 pos_encoding: str = 'sinusoidal',
                 projection_dim: int = 128, projection_hidden_dim: int = 256):
        """
        Args:
            input_dim: Input feature dimension
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            mlp_ratio: MLP expansion ratio
            dropout: Dropout rate
            num_classes: Number of classes
            pos_encoding: Type of positional encoding
            projection_dim: Output dimension of projection head
            projection_hidden_dim: Hidden dimension of projection head
        """
        self.input_dim = input_dim
        self.d_model = d_model
        self.num_classes = num_classes
        self.projection_dim = projection_dim

        # Input projection
        self.input_proj = Linear(input_dim, d_model)

        # CLS token (learnable parameter)
        self.cls_token = Parameter(np.random.randn(1, 1, d_model).astype(np.float32) * 0.02)

        # Positional encoding
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

        # Output normalization
        self.norm = LayerNorm(d_model)

        # Projection head (for contrastive learning)
        self.projection_head = ProjectionHead(d_model, projection_hidden_dim, projection_dim)

        # Classification head
        self.classifier = Linear(d_model, num_classes)

        # Cache for backward
        self.cache = {}

    def forward(self, x: np.ndarray, training: bool = True,
                return_embeddings: bool = True) -> tuple:
        """
        Forward pass

        Args:
            x: (batch, seq_len, input_dim)
            training: Whether in training mode
            return_embeddings: If True, return embeddings for contrastive loss

        Returns:
            If return_embeddings:
                (embeddings, logits)
            Else:
                logits
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
        cls_tokens = np.tile(self.cls_token.data, (batch_size, 1, 1))
        x = np.concatenate([cls_tokens, x], axis=1)

        # Add positional encoding
        x = self.pos_encoder.forward(x)
        self.cache['after_pos'] = x

        # Transformer layers
        for layer in self.layers:
            x = layer.forward(x)

        # Extract CLS token output (this is our representation)
        cls_out = x[:, 0, :]  # (batch, d_model)

        # Normalization
        features = self.norm.forward(cls_out)  # (batch, d_model)
        self.cache['features'] = features

        # Projection head for contrastive learning
        if return_embeddings:
            embeddings = self.projection_head.forward(features)  # (batch, projection_dim)
            self.cache['embeddings'] = embeddings
        else:
            embeddings = None

        # Classification logits
        logits = self.classifier.forward(features)  # (batch, num_classes)
        self.cache['logits'] = logits

        if return_embeddings:
            return embeddings, logits
        else:
            return logits

    def backward(self, grad_embeddings: np.ndarray = None,
                 grad_logits: np.ndarray = None) -> np.ndarray:
        """
        Backward pass

        Args:
            grad_embeddings: (batch, projection_dim) - gradient from contrastive loss
            grad_logits: (batch, num_classes) - gradient from classification loss

        Returns:
            grad_input: (batch, seq_len, input_dim)
        """
        features = self.cache['features']
        after_pos = self.cache['after_pos']
        batch_size, seq_len_plus_1, d_model = after_pos.shape

        # Accumulate gradients to features
        grad_features = np.zeros_like(features)

        # Backward through classification head
        if grad_logits is not None:
            grad_features += self.classifier.backward(grad_logits)

        # Backward through projection head
        if grad_embeddings is not None:
            grad_features += self.projection_head.backward(grad_embeddings)

        # Backward through normalization
        grad_cls_out = self.norm.backward(grad_features)

        # Backward through CLS token extraction
        grad_x = np.zeros((batch_size, seq_len_plus_1, d_model), dtype=np.float32)
        grad_x[:, 0, :] = grad_cls_out

        # Backward through transformer layers
        for layer in reversed(self.layers):
            grad_x = layer.backward(grad_x)

        # Backward through positional encoding
        grad_x = self.pos_encoder.backward(grad_x)

        # Split gradient: CLS token and input
        grad_cls_token = grad_x[:, 0:1, :]
        grad_input_proj = grad_x[:, 1:, :]

        # Accumulate gradient for CLS token
        self.cls_token.grad += grad_cls_token.sum(axis=0)

        # Backward through input projection
        grad_input = self.input_proj.backward(grad_input_proj)

        return grad_input

    def parameters(self):
        """Return all parameters"""
        params = [self.cls_token] + self.input_proj.parameters()
        for layer in self.layers:
            params.extend(layer.parameters())
        params.extend(self.norm.parameters())
        params.extend(self.projection_head.parameters())
        params.extend(self.classifier.parameters())
        return params

    def encoder_parameters(self):
        """Return only encoder parameters (no heads)"""
        params = [self.cls_token] + self.input_proj.parameters()
        for layer in self.layers:
            params.extend(layer.parameters())
        params.extend(self.norm.parameters())
        return params

    def projection_parameters(self):
        """Return only projection head parameters"""
        return self.projection_head.parameters()

    def classifier_parameters(self):
        """Return only classifier head parameters"""
        return self.classifier.parameters()

    def train(self):
        """Set to training mode"""
        self.pos_encoder.train()
        for layer in self.layers:
            layer.train()
        self.projection_head.train()

    def eval(self):
        """Set to evaluation mode"""
        self.pos_encoder.eval()
        for layer in self.layers:
            layer.eval()
        self.projection_head.eval()
