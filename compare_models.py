#!/usr/bin/env python3

import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from pretreat_points import load_data
from train_model import train_lstm_model, train_rnn_model

compared = "LSTM"

if __name__ == "__main__":
    np.random.seed(17)
    files = sorted(glob.glob("digits_3d/training_data/stroke_*_*.csv"))
    num_classes = 10

    # Raw data
    X_raw, y_raw = load_data(files, seq_len=50, normalize=False)
    X_raw_train, X_raw_test, y_raw_train_labels, y_raw_test_labels = train_test_split(X_raw, y_raw, test_size=0.2)
    # Normalized
    X_normalized, y_normalized = load_data(files, seq_len=50, normalize=True)
    X_normalized_train, X_normalized_test, y_normalized_train_labels, y_normalized_test_labels = train_test_split(X_normalized, y_normalized, test_size=0.2)
    # Augmented
    X_augmented, y_augmented = load_data(files, seq_len=50, normalize=True, sigma=0.02, scale_shift=0.1, max_angle=10)
    X_augmented_train, X_augmented_test, y_augmented_train_labels, y_augmented_test_labels = train_test_split(X_augmented, y_augmented, test_size=0.2)
    data_nature_label = ["Raw ", "Norm", "Augm"]

    Xs_train = [X_raw_train, X_normalized_train, X_augmented_train]
    ys_train = [y_raw_train_labels, y_normalized_train_labels, y_augmented_train_labels]
    Xs_test = [X_raw_test, X_normalized_test, X_augmented_test]
    ys_test = [y_raw_test_labels, y_normalized_test_labels, y_augmented_test_labels]

    hidden_sizes = [32, 64, 128]
    adams = ["False", "True "]
    nums_layers = [1, 2, 4, 6]

    accs = []
    models_names = []

    if compared == "LSTM":

        for idx_nature, nature in enumerate(data_nature_label):
            for _, hidden_size in enumerate(hidden_sizes):
                for adam, adam_label in enumerate(adams):
                    for _, num_layers in enumerate(nums_layers):
                        model_name = f"LSTM - {data_nature_label[idx_nature]} - hidden_size={hidden_size} - Adam={adam_label} - num_layers={num_layers}"
                        X_train, y_train_labels = Xs_train[idx_nature], ys_train[idx_nature]
                        X_test, y_test_labels = Xs_test[idx_nature], ys_test[idx_nature]
                        
                        lstm_model, accs, losses = train_lstm_model(
                            X_train, y_train_labels, batch_size=32, epochs=50, lr=0.001, hidden_size=hidden_size, test_split=0.2, adam=adam, num_layers=num_layers, verbose=False
                        )
                        _, y_pred_test = lstm_model.forward(X_test)
                        y_pred_test_labels = np.argmax(y_pred_test, axis=1)
                        acc = np.mean(y_pred_test_labels == y_test_labels)
                        accs.append(acc)
                        models_names.append(model_name)
                        print(f"{model_name}: {acc*100}%")
