import numpy as np

class LSTM:
    def __init__(self, input_size, hidden_size, output_size, lr=1e-3, adam=False, num_layers=1, seed=1):
        rng = np.random.RandomState(seed)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = lr
        self.num_layers = num_layers

        self.Wf = []
        self.Wi = []
        self.Wo = []
        self.Wc = []
        self.bf = []
        self.bi = []
        self.bo = []
        self.bc = []

        # Multiple layers initialization
        layer_input_size = input_size
        for l in range(num_layers):
            self.Wf.append(rng.randn(hidden_size, hidden_size + layer_input_size) * np.sqrt(1 / (hidden_size + layer_input_size)))
            self.Wi.append(rng.randn(hidden_size, hidden_size + layer_input_size) * np.sqrt(1 / (hidden_size + layer_input_size)))
            self.Wo.append(rng.randn(hidden_size, hidden_size + layer_input_size) * np.sqrt(1 / (hidden_size + layer_input_size)))
            self.Wc.append(rng.randn(hidden_size, hidden_size + layer_input_size) * np.sqrt(1 / (hidden_size + layer_input_size)))

            self.bf.append(np.zeros((hidden_size, 1)))
            self.bi.append(np.zeros((hidden_size, 1)))
            self.bo.append(np.zeros((hidden_size, 1)))
            self.bc.append(np.zeros((hidden_size, 1)))

            layer_input_size = hidden_size

        self.Wy = rng.randn(output_size, hidden_size) * np.sqrt(1 / hidden_size)
        self.by = np.zeros((output_size, 1))

        self.adam = adam
        if adam:
            self.beta1 = 0.9
            self.beta2 = 0.999
            self.eps = 1e-8
            self.t = 0

            # Adam parameters for each layer
            self.m_Wf, self.v_Wf = [np.zeros_like(w) for w in self.Wf], [np.zeros_like(w) for w in self.Wf]
            self.m_Wi, self.v_Wi = [np.zeros_like(w) for w in self.Wi], [np.zeros_like(w) for w in self.Wi]
            self.m_Wo, self.v_Wo = [np.zeros_like(w) for w in self.Wo], [np.zeros_like(w) for w in self.Wo]
            self.m_Wc, self.v_Wc = [np.zeros_like(w) for w in self.Wc], [np.zeros_like(w) for w in self.Wc]
            self.m_bf, self.v_bf = [np.zeros_like(b) for b in self.bf], [np.zeros_like(b) for b in self.bf]
            self.m_bi, self.v_bi = [np.zeros_like(b) for b in self.bi], [np.zeros_like(b) for b in self.bi]
            self.m_bo, self.v_bo = [np.zeros_like(b) for b in self.bo], [np.zeros_like(b) for b in self.bo]
            self.m_bc, self.v_bc = [np.zeros_like(b) for b in self.bc], [np.zeros_like(b) for b in self.bc]

            self.m_Wy = np.zeros_like(self.Wy); self.v_Wy = np.zeros_like(self.Wy)
            self.m_by = np.zeros_like(self.by); self.v_by = np.zeros_like(self.by)

    def forward(self, X):
        batch, seq_len, _ = X.shape
        h_layers = [np.zeros((batch, seq_len + 1, self.hidden_size)) for _ in range(self.num_layers)]
        c_layers = [np.zeros((batch, seq_len + 1, self.hidden_size)) for _ in range(self.num_layers)]
        cache_layers = [[] for _ in range(self.num_layers)]

        for t in range(seq_len):
            x_t = X[:, t, :]
            for l in range(self.num_layers):
                h_prev = h_layers[l][:, t, :]
                c_prev = c_layers[l][:, t, :]
                concat = np.hstack([h_prev, x_t])

                ft = self.sigmoid(concat @ self.Wf[l].T + self.bf[l].T)
                it = self.sigmoid(concat @ self.Wi[l].T + self.bi[l].T)
                ot = self.sigmoid(concat @ self.Wo[l].T + self.bo[l].T)
                ct_hat = np.tanh(concat @ self.Wc[l].T + self.bc[l].T)

                c_t = ft * c_prev + it * ct_hat
                h_t = ot * np.tanh(c_t)

                h_layers[l][:, t+1, :] = h_t
                c_layers[l][:, t+1, :] = c_t
                cache_layers[l].append((concat, ft, it, ot, ct_hat, c_prev, c_t))

                x_t = h_t  # input for next layer

        logits = h_layers[-1][:, -1, :] @ self.Wy.T + self.by.T
        self.cache_layers = cache_layers
        return (h_layers, c_layers), logits

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
        (h_layers, c_layers) = states
        batch, seq_len, _ = X.shape

        probs = self.softmax(logits)
        dy = (probs - y_true) / batch

        dWy = dy.T @ h_layers[-1][:, -1, :]
        dby = dy.T.sum(axis=1, keepdims=True)

        dh_next_layers = [np.zeros((batch, self.hidden_size)) for _ in range(self.num_layers)]
        dh_next_layers[-1] = dy @ self.Wy
        dc_next_layers = [np.zeros((batch, self.hidden_size)) for _ in range(self.num_layers)]

        # Initialize gradients for all LSTM parameters
        dWf = [0] * self.num_layers
        dWi = [0] * self.num_layers
        dWo = [0] * self.num_layers
        dWc = [0] * self.num_layers
        dbf = [0] * self.num_layers
        dbi = [0] * self.num_layers
        dbo = [0] * self.num_layers
        dbc = [0] * self.num_layers

        for t in reversed(range(seq_len)):
            dh_top = dh_next_layers[-1]
            for l in reversed(range(self.num_layers)):
                concat, ft, it, ot, ct_hat, ct_prev, ct = self.cache_layers[l][t]

                dh = dh_next_layers[l]
                do = dh * np.tanh(ct) * ot * (1 - ot)
                dct = dh * ot * (1 - np.tanh(ct)**2) + dc_next_layers[l]
                df = dct * ct_prev * ft * (1 - ft)
                di = dct * ct_hat * it * (1 - it)
                dch = dct * it * (1 - ct_hat**2)

                dWf[l] += df.T @ concat
                dWi[l] += di.T @ concat
                dWo[l] += do.T @ concat
                dWc[l] += dch.T @ concat

                dbf[l] += df.sum(axis=0).reshape(-1,1)
                dbi[l] += di.sum(axis=0).reshape(-1,1)
                dbo[l] += do.sum(axis=0).reshape(-1,1)
                dbc[l] += dch.sum(axis=0).reshape(-1,1)

                dconcat = (df @ self.Wf[l]) + (di @ self.Wi[l]) + (do @ self.Wo[l]) + (dch @ self.Wc[l])
                dh_next_layers[l] = dconcat[:, :self.hidden_size]
                dc_next_layers[l] = dct * ft

                if l > 0:
                    dh_next_layers[l-1] += dh_next_layers[l]

        # Clip gradients
        for l in range(self.num_layers):
            for g in [dWf[l], dWi[l], dWo[l], dWc[l]]:
                np.clip(g, -5, 5, out=g)

        np.clip(dWy, -5, 5, out=dWy)

        # Update weights with Adam or SGD
        if self.adam:
            self.t += 1
            for l in range(self.num_layers):
                self.adam_update(self.Wf[l], dWf[l], self.m_Wf[l], self.v_Wf[l], self.lr, self.t, self.beta1, self.beta2, self.eps)
                self.adam_update(self.Wi[l], dWi[l], self.m_Wi[l], self.v_Wi[l], self.lr, self.t, self.beta1, self.beta2, self.eps)
                self.adam_update(self.Wo[l], dWo[l], self.m_Wo[l], self.v_Wo[l], self.lr, self.t, self.beta1, self.beta2, self.eps)
                self.adam_update(self.Wc[l], dWc[l], self.m_Wc[l], self.v_Wc[l], self.lr, self.t, self.beta1, self.beta2, self.eps)
                self.adam_update(self.bf[l], dbf[l], self.m_bf[l], self.v_bf[l], self.lr, self.t, self.beta1, self.beta2, self.eps)
                self.adam_update(self.bi[l], dbi[l], self.m_bi[l], self.v_bi[l], self.lr, self.t, self.beta1, self.beta2, self.eps)
                self.adam_update(self.bo[l], dbo[l], self.m_bo[l], self.v_bo[l], self.lr, self.t, self.beta1, self.beta2, self.eps)
                self.adam_update(self.bc[l], dbc[l], self.m_bc[l], self.v_bc[l], self.lr, self.t, self.beta1, self.beta2, self.eps)

            self.adam_update(self.Wy, dWy, self.m_Wy, self.v_Wy, self.lr, self.t, self.beta1, self.beta2, self.eps)
            self.adam_update(self.by, dby, self.m_by, self.v_by, self.lr, self.t, self.beta1, self.beta2, self.eps)

        else:
            for l in range(self.num_layers):
                self.Wf[l] -= self.lr * dWf[l]
                self.Wi[l] -= self.lr * dWi[l]
                self.Wo[l] -= self.lr * dWo[l]
                self.Wc[l] -= self.lr * dWc[l]
                self.bf[l] -= self.lr * dbf[l]
                self.bi[l] -= self.lr * dbi[l]
                self.bo[l] -= self.lr * dbo[l]
                self.bc[l] -= self.lr * dbc[l]

            self.Wy -= self.lr * dWy
            self.by -= self.lr * dby

    def adam_update(self, param, grad, m, v, lr, t, beta1, beta2, eps):
        m[:] = beta1 * m + (1 - beta1) * grad
        v[:] = beta2 * v + (1 - beta2) * (grad * grad)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        param -= lr * m_hat / (np.sqrt(v_hat) + eps)

