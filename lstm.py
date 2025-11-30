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

class LSTM:
    def __init__(self, input_size, hidden_size, output_size, lr=1e-3, adam=False, seed=1):
        rng = np.random.RandomState(seed)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = lr

        self.Wf = rng.randn(hidden_size, hidden_size + input_size) * np.sqrt(1 / (hidden_size + input_size))
        self.Wi = rng.randn(hidden_size, hidden_size + input_size) * np.sqrt(1 / (hidden_size + input_size))
        self.Wo = rng.randn(hidden_size, hidden_size + input_size) * np.sqrt(1 / (hidden_size + input_size))
        self.Wc = rng.randn(hidden_size, hidden_size + input_size) * np.sqrt(1 / (hidden_size + input_size))

        self.bf = np.zeros((hidden_size, 1))
        self.bi = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))
        self.bc = np.zeros((hidden_size, 1))

        self.Wy = rng.randn(output_size, hidden_size) * np.sqrt(1 / hidden_size)
        self.by = np.zeros((output_size, 1))

        self.adam = adam
        if adam:
            self.beta1 = 0.9
            self.beta2 = 0.999
            self.eps = 1e-8
            self.t = 0

            self.m_Wf = np.zeros_like(self.Wf); self.v_Wf = np.zeros_like(self.Wf)
            self.m_Wi = np.zeros_like(self.Wi); self.v_Wi = np.zeros_like(self.Wi)
            self.m_Wo = np.zeros_like(self.Wo); self.v_Wo = np.zeros_like(self.Wo)
            self.m_Wc = np.zeros_like(self.Wc); self.v_Wc = np.zeros_like(self.Wc)
            self.m_Wy = np.zeros_like(self.Wy); self.v_Wy = np.zeros_like(self.Wy)

            self.m_bf = np.zeros_like(self.bf); self.v_bf = np.zeros_like(self.bf)
            self.m_bi = np.zeros_like(self.bi); self.v_bi = np.zeros_like(self.bi)
            self.m_bo = np.zeros_like(self.bo); self.v_bo = np.zeros_like(self.bo)
            self.m_bc = np.zeros_like(self.bc); self.v_bc = np.zeros_like(self.bc)
            self.m_by = np.zeros_like(self.by); self.v_by = np.zeros_like(self.by)

    def forward(self, X):
        batch, seq_len, _ = X.shape

        h = np.zeros((batch, seq_len + 1, self.hidden_size))
        c = np.zeros((batch, seq_len + 1, self.hidden_size))

        cache = []

        for t in range(seq_len):
            xt = X[:, t, :]                               
            ht_prev = h[:, t, :]
            ct_prev = c[:, t, :]

            concat = np.hstack([ht_prev, xt])             

            ft = self.sigmoid(concat @ self.Wf.T + self.bf.T)
            it = self.sigmoid(concat @ self.Wi.T + self.bi.T)
            ot = self.sigmoid(concat @ self.Wo.T + self.bo.T)
            ct_hat = np.tanh(concat @ self.Wc.T + self.bc.T)

            ct = ft * ct_prev + it * ct_hat
            ht = ot * np.tanh(ct)

            h[:, t+1, :] = ht
            c[:, t+1, :] = ct

            cache.append((concat, ft, it, ot, ct_hat, ct_prev, ct))

        logits = h[:, -1, :] @ self.Wy.T + self.by.T
        self.cache = cache
        return (h, c), logits

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def softmax(self, logits):
        exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp / np.sum(exp, axis=1, keepdims=True)

    def cross_entropy_loss(self, logits, y_true):
        probs = self.softmax(logits)
        loss = -np.sum(y_true * np.log(probs + 1e-12)) / y_true.shape[0]
        return loss, probs

    def bptt_update(self, X, states, logits, y_true):
        (h, c) = states
        cache = self.cache
        batch, seq_len, _ = X.shape

        probs = self.softmax(logits)
        dy = (probs - y_true) / batch

        dWy = dy.T @ h[:, -1, :]
        dby = dy.T.sum(axis=1, keepdims=True)

        dh_next = dy @ self.Wy
        dc_next = np.zeros_like(dh_next)

        dWf = dWi = dWo = dWc = 0
        dbf = dbi = dbo = dbc = 0

        for t in reversed(range(seq_len)):
            concat, ft, it, ot, ct_hat, ct_prev, ct = cache[t]

            dh = dh_next
            do = dh * np.tanh(ct) * ot * (1 - ot)
            dct = dh * ot * (1 - np.tanh(ct)**2) + dc_next
            df = dct * ct_prev * ft * (1 - ft)
            di = dct * ct_hat * it * (1 - it)
            dch = dct * it * (1 - ct_hat**2)

            dWf += df.T @ concat
            dWi += di.T @ concat
            dWo += do.T @ concat
            dWc += dch.T @ concat

            dbf += df.sum(axis=0).reshape(-1,1)
            dbi += di.sum(axis=0).reshape(-1,1)
            dbo += do.sum(axis=0).reshape(-1,1)
            dbc += dch.sum(axis=0).reshape(-1,1)

            dconcat = (df @ self.Wf) + (di @ self.Wi) + (do @ self.Wo) + (dch @ self.Wc)
            dh_next = dconcat[:, :self.hidden_size]
            dc_next = dct * ft

        for g in [dWf, dWi, dWo, dWc, dWy]:
            np.clip(g, -5, 5, out=g)

        if self.adam:
            self.t += 1
            self.adam_update(self.Wf, dWf, self.m_Wf, self.v_Wf, self.lr, self.t, self.beta1, self.beta2, self.eps)
            self.adam_update(self.Wi, dWi, self.m_Wi, self.v_Wi, self.lr, self.t, self.beta1, self.beta2, self.eps)
            self.adam_update(self.Wo, dWo, self.m_Wo, self.v_Wo, self.lr, self.t, self.beta1, self.beta2, self.eps)
            self.adam_update(self.Wc, dWc, self.m_Wc, self.v_Wc, self.lr, self.t, self.beta1, self.beta2, self.eps)
            self.adam_update(self.Wy, dWy, self.m_Wy, self.v_Wy, self.lr, self.t, self.beta1, self.beta2, self.eps)

            self.adam_update(self.bf, dbf, self.m_bf, self.v_bf, self.lr, self.t, self.beta1, self.beta2, self.eps)
            self.adam_update(self.bi, dbi, self.m_bi, self.v_bi, self.lr, self.t, self.beta1, self.beta2, self.eps)
            self.adam_update(self.bo, dbo, self.m_bo, self.v_bo, self.lr, self.t, self.beta1, self.beta2, self.eps)
            self.adam_update(self.bc, dbc, self.m_bc, self.v_bc, self.lr, self.t, self.beta1, self.beta2, self.eps)
            self.adam_update(self.by, dby, self.m_by, self.v_by, self.lr, self.t, self.beta1, self.beta2, self.eps)
        else:
            self.Wf -= self.lr * dWf
            self.Wi -= self.lr * dWi
            self.Wo -= self.lr * dWo
            self.Wc -= self.lr * dWc
            self.Wy -= self.lr * dWy

            self.bf -= self.lr * dbf
            self.bi -= self.lr * dbi
            self.bo -= self.lr * dbo
            self.bc -= self.lr * dbc
            self.by -= self.lr * dby
    
    def adam_update(self, param, grad, m, v, lr, t, beta1, beta2, eps):
        m[:] = beta1 * m + (1 - beta1) * grad
        v[:] = beta2 * v + (1 - beta2) * (grad * grad)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        param -= lr * m_hat / (np.sqrt(v_hat) + eps)


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

def train_lstm_model(X, y, batch_size=32, epochs=30, lr=0.001, hidden_size=64, test_split=0.2, adam=False, verbose=True):
    num_classes = 10
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_split, random_state=1)
    y_train = one_hot_encode(y_train, num_classes)
    n_train = X_train.shape[0]

    lstm = LSTM(input_size=X.shape[2], hidden_size=hidden_size, output_size=num_classes, lr=lr, adam=adam)

    losses, accs = [], []

    for epoch in range(1, epochs + 1):
        idx = np.random.permutation(n_train)
        X_shuffled = X_train[idx]
        y_shuffled = y_train[idx]

        epoch_loss = 0.0

        for i in range(0, n_train, batch_size):
            xb = X_shuffled[i:i + batch_size]
            yb = y_shuffled[i:i + batch_size]
            h, logits = lstm.forward(xb)
            loss, _ = lstm.cross_entropy_loss(logits, yb)
            epoch_loss += loss * xb.shape[0]
            lstm.bptt_update(xb, h, logits, yb)

        epoch_loss /= n_train
        losses.append(epoch_loss)

        # Evaluate on validation set
        _, y_val_pred = lstm.forward(X_val)
        y_val_pred_labels = np.argmax(y_val_pred, axis=1)
        acc = np.mean(y_val_pred_labels == y_val)
        accs.append(acc)

        if verbose:
            print(f"Epoch {epoch}/{epochs} - Loss: {epoch_loss:.6f} - Validation Accuracy: {acc:.4f}")

    return lstm, accs, losses


if __name__ == "__main__":
    files = sorted(glob.glob("digits_3d/training_data/stroke_*_*.csv"))
    X, y = load_data(files, seq_len=50)
    num_classes = 10
    X_train, X_test, y_train_labels, y_test_labels = train_test_split(X, y, test_size=0.2, random_state=42)

    lstm_model, accs, losses = train_lstm_model(
        X_train, y_train_labels, batch_size=32, epochs=30, lr=0.001, hidden_size=64, test_split=0.5, adam=True
    )
    torch.save(lstm_model, "lstm_model.pth")

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
    _, y_pred_test = lstm_model.forward(X_test)
    y_pred_test_labels = np.argmax(y_pred_test, axis=1)
    acc = np.mean(y_pred_test_labels == y_test_labels)
    print(f"Accuracy on test set: {acc}")
    cm = confusion_matrix(y_test_labels, y_pred_test_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix on Test Set")
    plt.show()
