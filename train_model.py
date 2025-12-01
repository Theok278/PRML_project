from rnn import RNN
from lstm import LSTM
from sklearn.model_selection import train_test_split
from pretreat_points import one_hot_encode
import numpy as np

def train_rnn_model(X, y, batch_size=32, epochs=30, lr=0.001, hidden_size=64, test_split=0.2, adam=False, num_layers= 1, verbose=True):
    num_classes = 10
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_split, random_state=1)
    y_train = one_hot_encode(y_train, num_classes)
    n_train = X_train.shape[0]

    rnn = RNN(input_size=X.shape[2], hidden_size=hidden_size, output_size=num_classes, lr=lr, adam=adam, num_layers=num_layers)

    losses, accs = [], []

    for epoch in range(1, epochs + 1):
        idx = np.random.permutation(n_train)
        X_shuffled = X_train[idx]
        y_shuffled = y_train[idx]

        epoch_loss = 0.0

        for i in range(0, n_train, batch_size):
            xb = X_shuffled[i:i + batch_size]
            yb = y_shuffled[i:i + batch_size]
            h_layers, logits = rnn.forward(xb)
            loss, _ = rnn.cross_entropy_loss(logits, yb)
            epoch_loss += loss * xb.shape[0]
            rnn.bptt_update(xb, h_layers, logits, yb)

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

def train_lstm_model(X, y, batch_size=32, epochs=30, lr=0.001, hidden_size=64, test_split=0.2, adam=False, num_layers=1, verbose=True):
    num_classes = 10
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_split, random_state=1)
    y_train = one_hot_encode(y_train, num_classes)
    n_train = X_train.shape[0]

    lstm = LSTM(input_size=X.shape[2], hidden_size=hidden_size, output_size=num_classes, lr=lr, adam=adam, num_layers=num_layers)

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

