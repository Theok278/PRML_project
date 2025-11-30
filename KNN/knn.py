import numpy as np


class KNNClassifier:
    """Simple k-NN classifier implemented with NumPy (brute-force)."""

    def __init__(self, k: int = 5):
        self.k = k
        self.x_train: np.ndarray | None = None
        self.y_train: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        # x: (N, D), y: (N,)
        self.x_train = x
        self.y_train = y

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.x_train is None or self.y_train is None:
            raise ValueError("Model not fitted.")
        # Compute squared Euclidean distances: (M, N)
        x2 = np.sum(x**2, axis=1, keepdims=True)  # (M, 1)
        y2 = np.sum(self.x_train**2, axis=1)  # (N,)
        cross = x @ self.x_train.T  # (M, N)
        dists = x2 - 2 * cross + y2  # (M, N)

        # Get k nearest indices
        idx = np.argpartition(dists, kth=self.k - 1, axis=1)[:, : self.k]  # (M, k)
        knn_labels = self.y_train[idx]  # (M, k)

        # Majority vote
        preds = []
        for row in knn_labels:
            vals, counts = np.unique(row, return_counts=True)
            preds.append(vals[np.argmax(counts)])
        return np.array(preds)
