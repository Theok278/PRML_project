import argparse
import json
import os
import numpy as np
from dataset import StrokeDataset

def standardize(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0, ddof=0)
    X_std = (X - mean) / std
    return X_std, mean, std

def rbf_kernel(X1, X2):
    gamma_val = 1.0 / (X1.shape[1] * X1.var())
    X1_sq = np.sum(X1**2, axis=1)[:, None]
    X2_sq = np.sum(X2**2, axis=1)[None, :]
    K = np.exp(-gamma_val * (X1_sq + X2_sq - 2 * np.dot(X1, X2.T)))
    return K

def train_rbf_svm_gd(X, y, C=1.0, lr=0.01, epochs=50):
    """
    Approximate binary RBF SVM with gradient descent on soft-margin primal.
    """
    n_samples = X.shape[0]

    alphas = np.zeros(n_samples)
    K = rbf_kernel(X, X)

    for epoch in range(epochs):
        for i in range(n_samples):
            margin = np.sum(alphas * y * K[:, i])
            grad = 1 - y[i] * margin
            if grad > 0:
                alphas[i] += lr * (1 - C * alphas[i])
            else:
                alphas[i] -= lr * (C * alphas[i])
        alphas = np.clip(alphas, 0, C)

    sv_mask = alphas > 1e-5
    b = np.mean(y[sv_mask] - np.sum(K[sv_mask][:, sv_mask] * (alphas[sv_mask] * y[sv_mask])[:, None], axis=0))
    return alphas, b, X, y

def predict_rbf_svm(X_test, model):
    alphas, b, X_train, y_train = model
    K_test = rbf_kernel(X_test, X_train)
    decision = np.dot(K_test, alphas * y_train) + b
    return decision 

def evaluate_rbf_kfold(x, y, C=1.0, k_folds=5, lr=0.01, epochs=50):
    """Manual K-fold cross-validation with OvR RBF SVM"""
    n_samples = x.shape[0]
    indices = np.arange(n_samples)
    rng = np.random.default_rng(42)
    rng.shuffle(indices)

    fold_size = n_samples // k_folds
    scores = []

    classes = np.unique(y)

    for k in range(k_folds):
        start = k * fold_size
        end = (k+1) * fold_size if k < k_folds - 1 else n_samples
        val_idx = indices[start:end]
        train_idx = np.concatenate([indices[:start], indices[end:]])

        X_train, y_train = x[train_idx], y[train_idx]
        X_val, y_val = x[val_idx], y[val_idx]

        # Standardize manually
        X_train_std, mean, std = standardize(X_train)
        X_val_std = (X_val - mean) / std

        # Train one-vs-rest classifiers
        models = {}
        for cls in classes:
            y_bin = np.where(y_train == cls, 1, -1)
            model = train_rbf_svm_gd(X_train_std, y_bin, C=C, lr=lr, epochs=epochs)
            models[cls] = model

        # Predict
        decision_matrix = np.zeros((X_val_std.shape[0], len(classes)))
        for idx, cls in enumerate(classes):
            decision_matrix[:, idx] = predict_rbf_svm(X_val_std, models[cls])

        y_pred = classes[np.argmax(decision_matrix, axis=1)]
        acc = np.mean(y_pred == y_val)
        scores.append(acc)

    mean_acc = np.mean(scores) * 100
    std_acc = np.std(scores) * 100
    return mean_acc, std_acc

def parsing():
    parser = argparse.ArgumentParser(description="Manual multi-class RBF SVM with k-fold CV.")
    parser.add_argument("--data_dir", type=str, default="../../digits_3d/training_data")
    parser.add_argument("--seq_len", type=int, default=50)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--k_folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="svm_results.json")
    return parser.parse_args()

def main() -> None:
    args = parsing()
    rng = np.random.default_rng(args.seed)
    ds = StrokeDataset(args.data_dir, seq_len=args.seq_len, normalization=True)
    x, y = ds.to_array()

    shuffled = rng.permutation(len(ds))
    x = x[shuffled]
    y = y[shuffled]

    mean_acc, std_acc = evaluate_rbf_kfold(
        x, y,
        C=args.C,
        k_folds=args.k_folds
    )

    print(f"RBF: {args.k_folds}-fold mean acc {mean_acc:.2f}% ± {std_acc:.2f}%")

    # Train final model on full dataset
    X_std, mean, std = standardize(x)
    classes = np.unique(y)
    final_models = {}
    for cls in classes:
        y_bin = np.where(y == cls, 1, -1)
        model = train_rbf_svm_gd(X_std, y_bin, C=args.C, lr=0.01, epochs=100)
        final_models[cls] = model

    results = {
        "seq_len": args.seq_len,
        "kernel": "rbf",
        "C": args.C,
        "k_folds": args.k_folds,
        "mean_acc": mean_acc,
        "std_acc": std_acc,
        "train_size": int(len(ds)),
        "scaler_mean": mean.tolist(),
        "scaler_std": std.tolist(),
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nSaved results to {args.output}")
    print("\nFinal model trained manually on full dataset.")


if __name__ == "__main__":
    main()
