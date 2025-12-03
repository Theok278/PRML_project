#!/usr/bin/env python3

import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from pretreat_points import load_data
from train_model import train_lstm_model

import numpy as np

def lstm_kfold(X, y, batch_size, hidden_size, num_layers, K=5, adam=True):
    n_samples = len(X)

    rng = np.random.default_rng(seed=17)
    indices = np.arange(n_samples)
    rng.shuffle(indices)

    folds = np.array_split(indices, K)
    fold_accuracies = []

    for fold in range(K):
        test_idx = folds[fold]
        train_idx = np.hstack([folds[i] for i in range(K) if i != fold])

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        lstm_model, _, _ = train_lstm_model(
            X_train, y_train, batch_size=batch_size, epochs=50, lr=0.001, hidden_size=hidden_size, test_split=0.2, adam=adam, num_layers=num_layers, verbose=False
        )

        _, y_pred = lstm_model.forward(X_test)
        y_pred_labels = np.argmax(y_pred, axis=1)
        acc = np.mean(y_pred_labels == y_test)

        fold_accuracies.append(acc)

    return np.array(fold_accuracies)


if __name__ == "__main__":
    np.random.seed(17)
    files = sorted(glob.glob("../digits_3d/training_data/stroke_*_*.csv"))
    num_classes = 10

    # Augmented
    X_augmented, y_augmented = load_data(files, seq_len=50, normalize=True, sigma=0.02, scale_shift=0.1, max_angle=10, deltas=False)
    # Adding dx, dy, dz
    X_delta, y_delta = load_data(files, seq_len=50, normalize=True, sigma=0.02, scale_shift=0.1, max_angle=10, deltas=True)

    Xs = [X_augmented, X_delta]
    ys = [y_augmented, y_delta]
    delta_labels = ["Augm", "Delt"]

    hidden_sizes = [96, 128, 160]
    num_layers = 2
    batch_sizes = [32, 24, 16, 8]
    adam = True

    accs = []
    std_accs = []
    models_names = []
    
    for hidden_size in hidden_sizes:
        for batch_size in batch_sizes:
            for idx_Xy, delta_label in enumerate(delta_labels):
                X = Xs[idx_Xy]
                y = ys[idx_Xy]
                model_name = f"LSTM hs={hidden_size} bs={batch_size} - {delta_label}"
                
                fold_accs = lstm_kfold(X, y, batch_size, hidden_size, num_layers, K=5, adam=adam)

                avg_acc = fold_accs.mean()
                std_acc = fold_accs.std()

                accs.append(avg_acc)
                std_accs.append(std_acc)
                models_names.append(model_name)

                print(f"{model_name}: acc={avg_acc*100:.3f}, std={std_acc:.4f}")

    print(accs)
    print(std_accs)
    print(models_names)
