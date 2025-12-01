import numpy as np

class RNN:
    def __init__(self, input_size, hidden_size, output_size, lr=1e-3, adam=False, num_layers=1, seed=1):
        rng = np.random.RandomState(seed)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.lr = lr

        self.Wxh = []
        self.Whh = []
        self.bh = []

        layer_input_size = input_size
        for l in range(num_layers):
            self.Wxh.append(rng.randn(hidden_size, layer_input_size) * np.sqrt(1 / layer_input_size))
            self.Whh.append(rng.randn(hidden_size, hidden_size) * np.sqrt(1 / hidden_size))
            self.bh.append(np.zeros((hidden_size, 1)))
            layer_input_size = hidden_size

        self.Why = rng.randn(output_size, hidden_size) * np.sqrt(1 / hidden_size)
        self.by = np.zeros((output_size, 1))

        self.adam = adam
        if adam:
            self.beta1 = 0.9
            self.beta2 = 0.999
            self.eps = 1e-8
            self.t = 0

            # Adam parameters per layer
            self.m_Wxh = [np.zeros_like(w) for w in self.Wxh]
            self.v_Wxh = [np.zeros_like(w) for w in self.Wxh]
            self.m_Whh = [np.zeros_like(w) for w in self.Whh]
            self.v_Whh = [np.zeros_like(w) for w in self.Whh]
            self.m_bh = [np.zeros_like(b) for b in self.bh]
            self.v_bh = [np.zeros_like(b) for b in self.bh]

            self.m_Why = np.zeros_like(self.Why)
            self.v_Why = np.zeros_like(self.Why)
            self.m_by = np.zeros_like(self.by)
            self.v_by = np.zeros_like(self.by)

    def forward(self, X):
        batch, seq_len, _ = X.shape
        h_layers = [np.zeros((batch, seq_len + 1, self.hidden_size)) for _ in range(self.num_layers)]

        for t in range(seq_len):
            x_t = X[:, t, :]
            for l in range(self.num_layers):
                h_prev = h_layers[l][:, t, :]
                pre = x_t.dot(self.Wxh[l].T) + h_prev.dot(self.Whh[l].T) + self.bh[l].T
                h_t = np.tanh(pre)
                h_layers[l][:, t + 1, :] = h_t
                x_t = h_t  # input for next layer

        logits = h_layers[-1][:, -1, :].dot(self.Why.T) + self.by.T
        self.h_layers = h_layers
        return h_layers, logits

    def softmax(self, logits):
        exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp / np.sum(exp, axis=1, keepdims=True)

    def cross_entropy_loss(self, logits, y_true):
        probs = self.softmax(logits)
        batch = y_true.shape[0]
        loss = -np.sum(y_true * np.log(probs + 1e-12)) / batch
        return loss, probs

    def bptt_update(self, X, h_layers, logits, y_true):
        batch, seq_len, _ = X.shape
        probs = self.softmax(logits)
        dy = (probs - y_true) / batch

        dWxh = [np.zeros_like(w) for w in self.Wxh]
        dWhh = [np.zeros_like(w) for w in self.Whh]
        dbh = [np.zeros_like(b) for b in self.bh]
        dWhy = np.zeros_like(self.Why)
        dby = np.zeros_like(self.by)

        dh_next_layers = [np.zeros((batch, self.hidden_size)) for _ in range(self.num_layers)]
        dh_next_layers[-1] = dy.dot(self.Why)

        for t in reversed(range(seq_len)):
            dh = dh_next_layers[-1]
            for l in reversed(range(self.num_layers)):
                ht = h_layers[l][:, t + 1, :]
                ht_prev = h_layers[l][:, t, :]
                dt = dh * (1 - ht**2)
                dbh[l] += dt.T.sum(axis=1, keepdims=True)
                xt = X[:, t, :] if l == 0 else h_layers[l-1][:, t + 1, :]
                dWxh[l] += dt.T.dot(xt)
                dWhh[l] += dt.T.dot(ht_prev)
                dh_next = dt.dot(self.Whh[l])
                if l > 0:
                    dh_next_layers[l-1] += dh_next
                dh = dh_next

        dWhy += dy.T.dot(h_layers[-1][:, -1, :])
        dby += dy.T.sum(axis=1, keepdims=True)

        # Clip gradients properly
        for l in range(self.num_layers):
            for grad in (dWxh[l], dWhh[l], dbh[l]):
                np.clip(grad, -5, 5, out=grad)
        np.clip(dWhy, -5, 5, out=dWhy)
        np.clip(dby, -5, 5, out=dby)

        # Update weights
        if self.adam:
            self.t += 1
            for l in range(self.num_layers):
                self.adam_update(self.Wxh[l], dWxh[l], self.m_Wxh[l], self.v_Wxh[l])
                self.adam_update(self.Whh[l], dWhh[l], self.m_Whh[l], self.v_Whh[l])
                self.adam_update(self.bh[l], dbh[l], self.m_bh[l], self.v_bh[l])
            self.adam_update(self.Why, dWhy, self.m_Why, self.v_Why)
            self.adam_update(self.by, dby, self.m_by, self.v_by)
        else:
            for l in range(self.num_layers):
                self.Wxh[l] -= self.lr * dWxh[l]
                self.Whh[l] -= self.lr * dWhh[l]
                self.bh[l] -= self.lr * dbh[l]
            self.Why -= self.lr * dWhy
            self.by -= self.lr * dby

    def adam_update(self, param, grad, m, v):
        m[:] = self.beta1 * m + (1 - self.beta1) * grad
        v[:] = self.beta2 * v + (1 - self.beta2) * (grad * grad)
        m_hat = m / (1 - self.beta1 ** self.t)
        v_hat = v / (1 - self.beta2 ** self.t)
        param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)




