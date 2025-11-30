import argparse
import json
import os
from typing import List, Tuple

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from dataset import StrokeDataset

def parse_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def evaluate_kernel(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, kernel: str,
                    c: float, gamma: str | float, degree: int) -> float:
    clf = make_pipeline(
        StandardScaler(),
        SVC(kernel=kernel, C=c, gamma=gamma, degree=degree)
    )
    clf.fit(x_train, y_train)
    acc = clf.score(x_val, y_val)
    return float(acc * 100.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="SVM baseline with kernel ablation for 3D digit strokes.")
    parser.add_argument("--data_dir", type=str, default="../../digits_3d/training_data")
    parser.add_argument("--seq_len", type=int, default=50)
    parser.add_argument("--kernels", type=str, default="linear,rbf,poly,sigmoid",
                        help="Comma-separated kernels to compare.")
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--gamma", type=str, default="scale")
    parser.add_argument("--degree", type=int, default=3, help="Degree for poly kernel.")
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="svm_results.json")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    kernels = parse_list(args.kernels)

    ds = StrokeDataset(args.data_dir, seq_len=args.seq_len, normalization=True)
    x, y = ds.to_array()

    shuffled = rng.permutation(len(ds))
    x = x[shuffled]
    y = y[shuffled]

    split = int(len(ds) * (1 - args.val_split))
    x_train, y_train = x[:split], y[:split]
    x_val, y_val = x[split:], y[split:]

    scores = {}
    best_kernel = None
    best_acc = -1.0

    for k in kernels:
        acc = evaluate_kernel(x_train, y_train, x_val, y_val, k, args.C, args.gamma, args.degree)
        scores[k] = acc
        print(f"Kernel={k}: val acc {acc:.2f}%")
        if acc > best_acc:
            best_acc = acc
            best_kernel = k

    # Train final model on full data with best kernel
    final_clf = make_pipeline(
        StandardScaler(),
        SVC(kernel=best_kernel, C=args.C, gamma=args.gamma, degree=args.degree)
    )
    final_clf.fit(x, y)

    results = {
        "seq_len": args.seq_len,
        "kernels": kernels,
        "C": args.C,
        "gamma": args.gamma,
        "degree": args.degree,
        "val_split": args.val_split,
        "scores": scores,
        "best_kernel": best_kernel,
        "best_acc": best_acc,
        "train_size": int(len(ds)),
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nBest kernel: {best_kernel} | val acc {best_acc:.2f}%")
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
