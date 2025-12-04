import numpy as np
from modules import Softmax


class CrossEntropyLoss:
    """Cross-entropy loss"""

    def __init__(self):
        self.softmax = Softmax(dim=-1)
        self.cache = {}

    def forward(self, logits: np.ndarray, targets: np.ndarray) -> float:
        """
        Args:
            logits: (batch, num_classes)
            targets: (batch,) - class indices
        Returns:
            loss: scalar
        """
        # Softmax
        probs = self.softmax.forward(logits)

        # Cross-entropy: -log(p[target])
        batch_size = logits.shape[0]
        log_probs = np.log(probs[np.arange(batch_size), targets] + 1e-10)
        loss = -log_probs.mean()

        # cache for backward
        self.cache['probs'] = probs
        self.cache['targets'] = targets
        self.cache['batch_size'] = batch_size

        return loss

    def backward(self) -> np.ndarray:
        """
        Returns:
            grad_logits: (batch, num_classes)
        """
        probs = self.cache['probs']
        targets = self.cache['targets']
        batch_size = self.cache['batch_size']

        # gradient of cross-entropy w.r.t. logits (after softmax)
        # d(-log(softmax(x)_i)) / dx_j = softmax(x)_j - delta_ij
        grad_logits = probs.copy()
        grad_logits[np.arange(batch_size), targets] -= 1.0
        grad_logits /= batch_size

        return grad_logits
