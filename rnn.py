#!/usr/bin/env python3

import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch

from pretreat_points import pretreat_points, resample_points

class RNN:
    def __init__(self, input_size, hidden_size, output_size, lr=1e-3, adam=False, seed=1):
        rng = np.random.RandomState(seed)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.Wxh = rng.randn(hidden_size, input_size) * np.sqrt(1 / input_size)
        self.Whh = rng.randn(hidden_size, hidden_size) * np.sqrt(1 / hidden_size) # Xavier/Glorot initialization
        self.Why = rng.randn(output_size, hidden_size) * np.sqrt(1 / hidden_size)
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))
        self.lr = lr

        self.adam = adam
        if adam:
            self.beta1 = 0.9
            self.beta2 = 0.999
            self.eps = 1e-8
            self.t = 0 

            self.m_Wxh = np.zeros_like(self.Wxh)
            self.v_Wxh = np.zeros_like(self.Wxh)

            self.m_Whh = np.zeros_like(self.Whh)
            self.v_Whh = np.zeros_like(self.Whh)

            self.m_Why = np.zeros_like(self.Why)
            self.v_Why = np.zeros_like(self.Why)

            self.m_bh = np.zeros_like(self.bh)
            self.v_bh = np.zeros_like(self.bh)

            self.m_by = np.zeros_like(self.by)
            self.v_by = np.zeros_like(self.by)

    def forward(self, X):
        batch, seq_len, _ = X.shape
        h = np.zeros((batch, seq_len + 1, self.hidden_size))
        for t in range(seq_len):
            xt = X[:, t, :].reshape(batch, -1)
            pre = xt.dot(self.Wxh.T) + h[:, t, :].dot(self.Whh.T) + self.bh.T
            h[:, t + 1, :] = np.tanh(pre)
        logits = h[:, -1, :].dot(self.Why.T) + self.by.T
        return h, logits

    def softmax(self, logits):
        exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp / np.sum(exp, axis=1, keepdims=True)

    def cross_entropy_loss(self, logits, y_true):
        probs = self.softmax(logits)
        batch = y_true.shape[0]
        loss = -np.sum(y_true * np.log(probs + 1e-12)) / batch
        return loss, probs
    
    def bptt_update(self, X, h, logits, y_true):
        batch, seq_len, _ = X.shape
        probs = self.softmax(logits)
        dy = (probs - y_true) / batch

        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dWhy = np.zeros_like(self.Why)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)

        h_last = h[:, -1, :].reshape(batch, self.hidden_size)
        dWhy += dy.T.dot(h_last)
        dby += dy.T.sum(axis=1, keepdims=True)

        dh_next = dy.dot(self.Why)

        for t in reversed(range(seq_len)):
            ht = h[:, t + 1, :]
            ht_prev = h[:, t, :]
            dt = dh_next * (1 - ht**2)
            dbh += dt.T.sum(axis=1, keepdims=True)
            xt = X[:, t, :].reshape(batch, -1)
            dWxh += dt.T.dot(xt)
            dWhh += dt.T.dot(ht_prev)
            dh_next = dt.dot(self.Whh)

        for grad in (dWxh, dWhh, dWhy, dbh, dby):
            np.clip(grad, -5, 5, out=grad)

        if self.adam:
            self.t += 1
            self.adam_update(self.Wxh, dWxh, self.m_Wxh, self.v_Wxh)
            self.adam_update(self.Whh, dWhh, self.m_Whh, self.v_Whh)
            self.adam_update(self.Why, dWhy, self.m_Why, self.v_Why)
            self.adam_update(self.bh,  dbh,  self.m_bh,  self.v_bh)
            self.adam_update(self.by,  dby,  self.m_by,  self.v_by)

        else:
            self.Wxh -= self.lr * dWxh
            self.Whh -= self.lr * dWhh
            self.Why -= self.lr * dWhy
            self.bh -= self.lr * dbh
            self.by -= self.lr * dby

    def adam_update(self, param, grad, m, v):
        m[:] = self.beta1 * m + (1 - self.beta1) * grad
        v[:] = self.beta2 * v + (1 - self.beta2) * (grad * grad)
        m_hat = m / (1 - self.beta1 ** self.t)
        v_hat = v / (1 - self.beta2 ** self.t)
        param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

def load_data(files, seq_len=50):
    X_list, y_list = [], []
    for f in files:
        df = pd.read_csv(f, header=None)
        pts = df.to_numpy()
        mean = pts.mean(axis=0)
        std = pts.std(axis=0)
        pts = (pts - mean) / std
        pts = resample_points(pts, seq_len)
        X_list.append(pts.astype(np.float32))

        label = int(os.path.basename(f).split("_")[1])
        y_list.append(label)
    X = np.stack(X_list)
    y = np.array(y_list)
    return X, y

def one_hot_encode(y, num_classes):
    return np.eye(num_classes)[y]

def train_rnn_model(X, y, batch_size=32, epochs=30, lr=0.001, hidden_size=64, test_split=0.2, adam=False, verbose=True):
    num_classes = 10
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_split, random_state=1)
    y_train = one_hot_encode(y_train, num_classes)
    n_train = X_train.shape[0]

    rnn = RNN(input_size=X.shape[2], hidden_size=hidden_size, output_size=num_classes, lr=lr, adam=adam)

    losses, accs = [], []

    for epoch in range(1, epochs + 1):
        idx = np.random.permutation(n_train)
        X_shuffled = X_train[idx]
        y_shuffled = y_train[idx]

        epoch_loss = 0.0

        for i in range(0, n_train, batch_size):
            xb = X_shuffled[i:i + batch_size]
            yb = y_shuffled[i:i + batch_size]
            h, logits = rnn.forward(xb)
            loss, _ = rnn.cross_entropy_loss(logits, yb)
            epoch_loss += loss * xb.shape[0]
            rnn.bptt_update(xb, h, logits, yb)

        epoch_loss /= n_train
        losses.append(epoch_loss)

        # Evaluate on validation set
        _, y_val_pred = rnn.forward(X_val)
        y_val_pred_labels = np.argmax(y_val_pred, axis=1)
        acc = np.mean(y_val_pred_labels == y_val)
        accs.append(acc)

        if verbose:
            print(f"Epoch {epoch}/{epochs} - Loss: {epoch_loss:.6f} - Validation Accuracy: {acc:.4f}")

    return rnn, accs, losses


if __name__ == "__main__":
    files = sorted(glob.glob("digits_3d/training_data/stroke_*_*.csv"))
    X, y = load_data(files, seq_len=50)
    num_classes = 10
    X_train, X_test, y_train_labels, y_test_labels = train_test_split(X, y, test_size=0.2, random_state=42)

    rnn_model, accs, losses = train_rnn_model(
        X_train, y_train_labels, batch_size=32, epochs=100, lr=0.001, hidden_size=128, test_split=0.5, adam=True
    )
    torch.save(rnn_model, "rnn_model.pth")

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
