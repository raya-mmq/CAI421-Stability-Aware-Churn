"""
CAI 421 - Checkpoint 2 - MLP baseline (Student 5)

A fully connected network for binary churn prediction, written in NumPy so that
the forward pass and backpropagation are explicit and match the equations and
the induction proof in the Method section exactly:

    a^(0) = x
    z^(l) = W^(l) a^(l-1) + b^(l),   a^(l) = ReLU(z^(l))        l = 1..L-1
    z^(L) = W^(L) a^(L-1) + b^(L),   y_hat = sigmoid(z^(L))
    loss  = binary cross-entropy (mean over the mini-batch) + (lambda/2)*sum ||W||^2

Backpropagation (proved by backward induction on l):
    delta^(L) = (y_hat - y) / B
    delta^(l) = (W^(l+1)^T delta^(l+1)) * ReLU'(z^(l))
    dL/dW^(l) = delta^(l) a^(l-1)^T (+ lambda W^(l)),   dL/db^(l) = sum_batch delta^(l)

Rows are samples, so in code W^(l) has shape (n_{l-1}, n_l) and z = a @ W + b.
"""
import numpy as np

EPS = 1e-12


def sigmoid(z):
    # numerically stable logistic function
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def bce(y_hat, y):
    y_hat = np.clip(y_hat, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)))


class MLP:
    def __init__(self, n_in, hidden=(64, 32), weight_decay=0.0, seed=42):
        self.sizes = [n_in, *hidden, 1]
        self.weight_decay = weight_decay
        rng = np.random.default_rng(seed)
        self.W, self.b = [], []
        for l in range(1, len(self.sizes)):
            fan_in, fan_out = self.sizes[l - 1], self.sizes[l]
            # He initialisation for ReLU layers, Xavier/Glorot for the sigmoid output layer
            std = np.sqrt(2.0 / fan_in) if l < len(self.sizes) - 1 else np.sqrt(1.0 / fan_in)
            self.W.append(rng.normal(0.0, std, size=(fan_in, fan_out)))
            self.b.append(np.zeros(fan_out))

    # ---------- parameters as a flat list (used by the optimiser and grad check) ----------
    @property
    def params(self):
        return [*self.W, *self.b]

    def set_params(self, params):
        k = len(self.W)
        self.W = [p.copy() for p in params[:k]]
        self.b = [p.copy() for p in params[k:]]

    # ---------- forward ----------
    def forward(self, X):
        """Returns y_hat (shape (B,)) and the cache [(a^(l-1), z^(l)) for l = 1..L]."""
        a = X
        cache = []
        L = len(self.W)
        for l in range(L):
            z = a @ self.W[l] + self.b[l]
            cache.append((a, z))
            a = np.maximum(z, 0.0) if l < L - 1 else sigmoid(z)
        return a.ravel(), cache

    def predict_proba(self, X):
        return self.forward(np.asarray(X, dtype=np.float64))[0]

    def loss(self, X, y):
        y_hat, _ = self.forward(X)
        reg = 0.5 * self.weight_decay * sum(float(np.sum(W * W)) for W in self.W)
        return bce(y_hat, y) + reg

    # ---------- backward (backpropagation) ----------
    def backward(self, y_hat, y, cache):
        """Gradients of the mean BCE (+ L2) w.r.t. every W^(l), b^(l)."""
        B = y.shape[0]
        L = len(self.W)
        gW, gb = [None] * L, [None] * L
        # base case: output layer, sigmoid + BCE  ->  delta^(L) = (y_hat - y) / B
        delta = ((y_hat - y) / B).reshape(-1, 1)
        for l in range(L - 1, -1, -1):
            a_prev, _ = cache[l]
            gW[l] = a_prev.T @ delta + self.weight_decay * self.W[l]
            gb[l] = delta.sum(axis=0)
            if l > 0:
                # inductive step: delta^(l-1) = (delta^(l) W^(l)^T) * ReLU'(z^(l-1))
                _, z_prev = cache[l - 1]
                delta = (delta @ self.W[l].T) * (z_prev > 0)
        return [*gW, *gb]


class Adam:
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        new = []
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            m_hat = self.m[i] / (1 - self.b1 ** self.t)
            v_hat = self.v[i] / (1 - self.b2 ** self.t)
            new.append(p - self.lr * m_hat / (np.sqrt(v_hat) + self.eps))
        return new


def train(X_tr, y_tr, X_va, y_va, hidden=(64, 32), lr=1e-3, batch_size=64,
          weight_decay=0.0, max_epochs=300, patience=20, seed=42):
    """Mini-batch Adam with early stopping on validation BCE; restores the best epoch."""
    X_tr = np.asarray(X_tr, dtype=np.float64)
    X_va = np.asarray(X_va, dtype=np.float64)
    y_tr = np.asarray(y_tr, dtype=np.float64)
    y_va = np.asarray(y_va, dtype=np.float64)

    model = MLP(X_tr.shape[1], hidden, weight_decay, seed)
    opt = Adam(model.params, lr=lr)
    rng = np.random.default_rng(seed + 1)          # shuffling stream, separate from init
    history = {"train_loss": [], "val_loss": []}
    best_val, best_params, best_epoch, wait = np.inf, model.params, 0, 0

    for epoch in range(1, max_epochs + 1):
        order = rng.permutation(len(X_tr))
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            y_hat, cache = model.forward(X_tr[idx])
            grads = model.backward(y_hat, y_tr[idx], cache)
            model.set_params(opt.step(model.params, grads))

        tr_loss = bce(model.predict_proba(X_tr), y_tr)
        va_loss = bce(model.predict_proba(X_va), y_va)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)

        if va_loss < best_val - 1e-5:
            best_val, best_params, best_epoch, wait = va_loss, model.params, epoch, 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.set_params(best_params)
    return model, history, best_epoch
