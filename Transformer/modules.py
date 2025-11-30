import numpy as np
import math
from typing import Tuple, List, Optional


class Parameter:
    """Wrapper for parameters that need gradients"""
    def __init__(self, data: np.ndarray):
        self.data = data.astype(np.float32)
        self.grad = np.zeros_like(data, dtype=np.float32)

    def zero_grad(self):
        self.grad.fill(0)


class PositionalEncoding:
    """Sinusoidal positional encoding"""

    def __init__(self, d_model: int, max_len: int = 512):
        self.d_model = d_model

        # pre-compute positional encodings (max_len, d_model)
        pe = np.zeros((max_len, d_model), dtype=np.float32)
        position = np.arange(0, max_len, dtype=np.float32)[:, np.newaxis]

        # div_term: (d_model/2,)
        div_term = np.exp(
            np.arange(0, d_model, 2, dtype=np.float32) *
            (-np.log(10000.0) / d_model)
        )

        # even dimensions: sin, odd dimensions: cos
        pe[:, 0::2] = np.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = np.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = np.cos(position * div_term)

        self.pe = pe

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Add positional encoding to input

        Args:
            x: (batch, seq_len, d_model)
        Returns:
            x + positional encoding: (batch, seq_len, d_model)
        """
        seq_len = x.shape[1]
        # broadcasting: (batch, seq_len, d_model) + (seq_len, d_model)
        return x + self.pe[:seq_len, :]

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """positional encoding is fixed, gradient passes through"""
        return grad_output


class Linear:
    """Linear layer with forward and backward pass"""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        self.in_features = in_features
        self.out_features = out_features

        # xavier initialization
        std = np.sqrt(2.0 / (in_features + out_features))
        self.weight = Parameter(np.random.randn(in_features, out_features).astype(np.float32) * std)

        if bias:
            self.bias = Parameter(np.zeros(out_features, dtype=np.float32))
        else:
            self.bias = None

        # cache for backward pass
        self.cache_input = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass: y = xW + b

        Args:
            x: (batch, ..., in_features)
        Returns:
            y: (batch, ..., out_features)
        """
        self.cache_input = x
        output = x @ self.weight.data

        if self.bias is not None:
            output = output + self.bias.data

        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass

        Args:
            grad_output: (batch, ..., out_features)
        Returns:
            grad_input: (batch, ..., in_features)
        """
        # gradient w.r.t weight: x^T @ grad_output
        # need to reshape for batched matrix multiplication
        original_shape = self.cache_input.shape
        x_2d = self.cache_input.reshape(-1, self.in_features)  
        # (batch*..., in_features)
        grad_2d = grad_output.reshape(-1, self.out_features)    
        # (batch*..., out_features)

        self.weight.grad += x_2d.T @ grad_2d  
        # (in_features, out_features)

        # gradient w.r.t bias
        if self.bias is not None:
            self.bias.grad += grad_2d.sum(axis=0)

        # gradient w.r.t input
        grad_input = grad_output @ self.weight.data.T

        return grad_input

    def parameters(self):
        if self.bias is not None:
            return [self.weight, self.bias]
        return [self.weight]


class LayerNorm:
    """Layer Normalization with forward and backward pass"""

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        self.normalized_shape = normalized_shape
        self.eps = eps

        self.gamma = Parameter(np.ones(normalized_shape, dtype=np.float32))
        self.beta = Parameter(np.zeros(normalized_shape, dtype=np.float32))

        # cache for backward
        self.cache = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        y = gamma * (x - mean) / sqrt(var + eps) + beta
        """
        # compute along last dimension
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)

        x_norm = (x - mean) / np.sqrt(var + self.eps)
        output = self.gamma.data * x_norm + self.beta.data

        # cache for backward
        self.cache = {
            'x': x,
            'x_norm': x_norm,
            'mean': mean,
            'var': var,
            'std': np.sqrt(var + self.eps)
        }

        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """backward pass for LayerNorm"""
        x = self.cache['x']
        x_norm = self.cache['x_norm']
        std = self.cache['std']

        N = self.normalized_shape

        # gradients w.r.t gamma and beta
        self.gamma.grad += (grad_output * x_norm).sum(axis=tuple(range(len(grad_output.shape) - 1)))
        self.beta.grad += grad_output.sum(axis=tuple(range(len(grad_output.shape) - 1)))

        # gradient w.r.t input (complex due to mean and variance)
        grad_x_norm = grad_output * self.gamma.data

        grad_var = (grad_x_norm * (x - self.cache['mean']) * -0.5 * (std ** -3)).sum(axis=-1, keepdims=True)
        grad_mean = (grad_x_norm * (-1.0 / std)).sum(axis=-1, keepdims=True) + \
                    grad_var * (x - self.cache['mean']).sum(axis=-1, keepdims=True) * (-2.0 / N)

        grad_input = grad_x_norm / std + \
                     grad_var * 2.0 * (x - self.cache['mean']) / N + \
                     grad_mean / N

        return grad_input

    def parameters(self):
        return [self.gamma, self.beta]


class GELU:
    """GELU activation with forward and backward pass"""

    def __init__(self):
        self.cache_input = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """GELU(x) = x * Phi(x) where Phi is CDF of standard normal"""
        self.cache_input = x
        # approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """backward pass for GELU"""
        x = self.cache_input

        # derivative of GELU (approximation)
        tanh_arg = np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)
        tanh_out = np.tanh(tanh_arg)

        dtanh = 1.0 - tanh_out ** 2
        darg = np.sqrt(2.0 / np.pi) * (1.0 + 3.0 * 0.044715 * x ** 2)

        grad_gelu = 0.5 * (1.0 + tanh_out) + 0.5 * x * dtanh * darg

        return grad_output * grad_gelu


class Softmax:
    """Softmax with forward and backward pass"""

    def __init__(self, dim: int = -1):
        self.dim = dim
        self.cache_output = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """numerically stable softmax"""
        # subtract max for numerical stability
        x_max = x.max(axis=self.dim, keepdims=True)
        exp_x = np.exp(x - x_max)
        output = exp_x / exp_x.sum(axis=self.dim, keepdims=True)

        self.cache_output = output
        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        backward for softmax
        For softmax s, jacobian is: J_ij = s_i * (delta_ij - s_j)
        grad_input_i = sum_j (grad_output_j * J_ji)
                     = sum_j (grad_output_j * s_j * (delta_ji - s_i))
                     = grad_output_i * s_i - s_i * sum_j(grad_output_j * s_j)
        """
        s = self.cache_output

        # sum_j(grad_output_j * s_j)
        sum_term = (grad_output * s).sum(axis=self.dim, keepdims=True)

        grad_input = s * (grad_output - sum_term)

        return grad_input


class Dropout:
    """Dropout with forward and backward pass"""

    def __init__(self, p: float = 0.1):
        self.p = p
        self.mask = None
        self.training = True

    def forward(self, x: np.ndarray) -> np.ndarray:
        if not self.training or self.p == 0:
            return x

        # create dropout mask
        self.mask = (np.random.rand(*x.shape) > self.p).astype(np.float32)
        # scale by 1/(1-p) to maintain expected value
        return x * self.mask / (1.0 - self.p)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if not self.training or self.p == 0:
            return grad_output

        return grad_output * self.mask / (1.0 - self.p)

    def train(self):
        self.training = True

    def eval(self):
        self.training = False


class MultiHeadAttention:
    """Multi-head self-attention with forward and backward pass"""

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        assert d_model % nhead == 0

        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scale = self.head_dim ** -0.5

        self.qkv = Linear(d_model, d_model * 3, bias=True)
        self.out_proj = Linear(d_model, d_model, bias=True)

        self.attn_dropout = Dropout(dropout)
        self.out_dropout = Dropout(dropout)

        self.softmax = Softmax(dim=-1)

        # cache for backward
        self.cache = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            output: (batch, seq_len, d_model)
        """
        B, N, D = x.shape

        # QKV projection
        qkv = self.qkv.forward(x)  # (B, N, 3*d_model)
        q, k, v = np.split(qkv, 3, axis=-1)

        # Reshape for multi-head: (B, N, d_model) -> (B, nhead, N, head_dim)
        def reshape_heads(t):
            return t.reshape(B, N, self.nhead, self.head_dim).transpose(0, 2, 1, 3)
        q = reshape_heads(q)  # (B, H, N, Dk)
        k = reshape_heads(k)
        v = reshape_heads(v)

        # Attention scores
        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale  # (B, H, N, N)
        attn = self.softmax.forward(scores)
        attn = self.attn_dropout.forward(attn)

        # Apply attention to values
        context = attn @ v  # (B, H, N, Dk)

        # Reshape back: (B, H, N, Dk) -> (B, N, d_model)
        context = context.transpose(0, 2, 1, 3).reshape(B, N, D)

        # Output projection
        output = self.out_proj.forward(context)
        output = self.out_dropout.forward(output)

        # Cache for backward
        self.cache = {
            'q': q, 'k': k, 'v': v,
            'scores': scores,
            'attn': attn,
            'context': context,
            'B': B, 'N': N, 'D': D
        }

        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        B, N, D = self.cache['B'], self.cache['N'], self.cache['D']
        q, k, v = self.cache['q'], self.cache['k'], self.cache['v']
        attn = self.cache['attn']

        # Backward through output projection
        grad_output = self.out_dropout.backward(grad_output)
        grad_context = self.out_proj.backward(grad_output)

        # Reshape gradient: (B, N, D) -> (B, H, N, Dk)
        grad_context = grad_context.reshape(B, N, self.nhead, self.head_dim).transpose(0, 2, 1, 3)

        # Backward through attention application: context = attn @ v
        grad_attn = grad_context @ v.transpose(0, 1, 3, 2)  # (B, H, N, N)
        grad_v = attn.transpose(0, 1, 3, 2) @ grad_context  # (B, H, N, Dk)

        # Backward through dropout and softmax
        grad_attn = self.attn_dropout.backward(grad_attn)
        grad_scores = self.softmax.backward(grad_attn)

        # Backward through scaled dot-product: scores = q @ k^T * scale
        grad_scores_scaled = grad_scores * self.scale
        grad_q = grad_scores_scaled @ k  # (B, H, N, Dk)
        grad_k = grad_scores_scaled.transpose(0, 1, 3, 2) @ q  # (B, H, N, Dk)

        # Reshape back: (B, H, N, Dk) -> (B, N, D)
        def reshape_back(t):
            return t.transpose(0, 2, 1, 3).reshape(B, N, D)

        grad_q = reshape_back(grad_q)
        grad_k = reshape_back(grad_k)
        grad_v = reshape_back(grad_v)

        # Concatenate gradients
        grad_qkv = np.concatenate([grad_q, grad_k, grad_v], axis=-1)

        # Backward through QKV projection
        grad_input = self.qkv.backward(grad_qkv)

        return grad_input

    def parameters(self):
        return self.qkv.parameters() + self.out_proj.parameters()

    def train(self):
        self.attn_dropout.train()
        self.out_dropout.train()

    def eval(self):
        self.attn_dropout.eval()
        self.out_dropout.eval()