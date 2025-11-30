import argparse
import json
import os
from typing import List

import numpy as np

from dataset import StrokeDataset
from knn import KNNClassifier


def kfold_indices(n_samples: int, n_splits: int, rng: np.random.Generator) -> List[np.ndarray]:
    """Generate list of index arrays for each fold."""
    indices = rng.permutation(n_samples)
    folds = np.array_split(indices, n_splits)
    return folds


def evaluate_k(x: np.ndarray, y: np.ndarray, folds: List[np.ndarray], k: int) -> float:
    """Return mean accuracy across folds for a given k."""
    accs = []
    for i in range(len(folds)):
        val_idx = folds[i]
        train_idx = np.hstack([folds[j] for j in range(len(folds)) if j != i])

        clf = KNNClassifier(k=k)
        clf.fit(x[train_idx], y[train_idx])
        preds = clf.predict(x[val_idx])
        acc = (preds == y[val_idx]).mean()
        accs.append(acc)
    return float(np.mean(accs) * 100.0)


def parse_k_list(k_list_str: str) -> List[int]:
    return [int(k.strip()) for k in k_list_str.split(",") if k.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="k-NN classifier for 3D digit strokes with 5-fold CV grid search on k.")
    parser.add_argument("--data_dir", type=str, default="../../digits_3d/training_data")
    parser.add_argument("--seq_len", type=int, default=50)
    parser.add_argument("--k_list", type=str, default="1,3,5,7,9", help="Comma-separated k candidates.")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="results.json")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    ks = parse_k_list(args.k_list)

    ds = StrokeDataset(args.data_dir, seq_len=args.seq_len, normalization=True)
    x, y = ds.to_array()  # x shape: (N, seq_len*3)

    folds = kfold_indices(len(ds), args.folds, rng)

    scores = {}
    best_k = None
    best_acc = -1.0
    for k in ks:
        acc = evaluate_k(x, y, folds, k)
        scores[k] = acc
        print(f"k={k}: mean 5-fold acc {acc:.2f}%")
        if acc > best_acc:
            best_acc = acc
            best_k = k

    # Fit on full data with best k
    clf = KNNClassifier(k=best_k)
    clf.fit(x, y)

    results = {
        "seq_len": args.seq_len,
        "folds": args.folds,
        "k_list": ks,
        "cv_scores": scores,
        "best_k": best_k,
        "best_acc": best_acc,
        "train_size": int(len(ds)),
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nBest k={best_k} with mean acc {best_acc:.2f}%")
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
