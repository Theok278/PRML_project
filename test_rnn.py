#!/usr/bin/env python3

import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from pretreat_points import load_data
from train_model import train_rnn_model
from save_model import save_rnn_model

if __name__ == "__main__":
    np.random.seed(42)
    files = sorted(glob.glob("digits_3d/training_data/stroke_*_*.csv"))
    X, y = load_data(files, seq_len=50, normalize=True, sigma=0.02, scale_shift=0.1, max_angle=10)
    num_classes = 10
    X_train, X_test, y_train_labels, y_test_labels = train_test_split(X, y, test_size=0.2)

    rnn_model, accs, losses = train_rnn_model(
        X_train, y_train_labels, batch_size=32, epochs=100, lr=0.001, hidden_size=128, test_split=0.5, adam=True, num_layers=4
    )
    save_rnn_model(rnn_model, "rnn_model.npz")

    # Plot training
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Test Accuracy', color='blue')
    ax1.plot(accs, color='blue', label='Test Accuracy')
    ax1.tick_params(axis='y', labelcolor='blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Loss', color='red')
    ax2.plot(losses, color='red', label='Loss')
    ax2.tick_params(axis='y', labelcolor='red')

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='best')
    plt.title('LSTM Training Loss and Test Accuracy')
    plt.show()

    # Confusion matrix
    _, y_pred_test = rnn_model.forward(X_test)
    y_pred_test_labels = np.argmax(y_pred_test, axis=1)
    acc = np.mean(y_pred_test_labels == y_test_labels)
    print(f"Accuracy on test set: {acc}")
    cm = confusion_matrix(y_test_labels, y_pred_test_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix on Test Set")
    plt.show()