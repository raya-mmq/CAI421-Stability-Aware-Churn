"""NumPy MLP implementation aligned with the team's Checkpoint 2 baseline."""
import numpy as np

EPS = 1e-12

def sigmoid(z):
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out

def bce(y_hat, y):
    y_hat = np.clip(y_hat, EPS, 1.0 - EPS)
    return float(-np.mean(y * np.log(y_hat) + (1.0-y) * np.log(1.0-y_hat)))

class MLP:
    def __init__(self, n_in, hidden=(32,), weight_decay=1e-4, seed=42):
        self.sizes = [n_in, *hidden, 1]
        self.weight_decay = weight_decay
        rng = np.random.default_rng(seed)
        self.W, self.b = [], []
        for layer in range(1, len(self.sizes)):
            fan_in, fan_out = self.sizes[layer-1], self.sizes[layer]
            std = np.sqrt(2.0/fan_in) if layer < len(self.sizes)-1 else np.sqrt(1.0/fan_in)
            self.W.append(rng.normal(0.0, std, size=(fan_in, fan_out)))
            self.b.append(np.zeros(fan_out))

    @property
    def params(self):
        return [*self.W, *self.b]

    def set_params(self, params):
        k = len(self.W)
        self.W = [p.copy() for p in params[:k]]
        self.b = [p.copy() for p in params[k:]]

    def forward(self, X):
        a = np.asarray(X, dtype=np.float64)
        cache = []
        for layer in range(len(self.W)):
            z = a @ self.W[layer] + self.b[layer]
            cache.append((a, z))
            a = np.maximum(z, 0.0) if layer < len(self.W)-1 else sigmoid(z)
        return a.ravel(), cache

    def predict_proba(self, X):
        return self.forward(np.asarray(X, dtype=np.float64))[0]

    def backward(self, y_hat, y, cache):
        B = y.shape[0]
        L = len(self.W)
        gW, gb = [None]*L, [None]*L
        delta = ((y_hat-y)/B).reshape(-1, 1)
        for layer in range(L-1, -1, -1):
            a_prev, _ = cache[layer]
            gW[layer] = a_prev.T @ delta + self.weight_decay*self.W[layer]
            gb[layer] = delta.sum(axis=0)
            if layer > 0:
                _, z_prev = cache[layer-1]
                delta = (delta @ self.W[layer].T) * (z_prev > 0)
        return [*gW, *gb]

class Adam:
    def __init__(self, params, lr=0.003, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        new = []
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = self.b1*self.m[i] + (1-self.b1)*g
            self.v[i] = self.b2*self.v[i] + (1-self.b2)*g*g
            mh = self.m[i]/(1-self.b1**self.t)
            vh = self.v[i]/(1-self.b2**self.t)
            new.append(p-self.lr*mh/(np.sqrt(vh)+self.eps))
        return new

def train(X_tr, y_tr, X_va, y_va, hidden=(32,), lr=0.003, batch_size=128,
          weight_decay=1e-4, max_epochs=300, patience=20, seed=42):
    X_tr = np.asarray(X_tr, dtype=np.float64)
    X_va = np.asarray(X_va, dtype=np.float64)
    y_tr = np.asarray(y_tr, dtype=np.float64)
    y_va = np.asarray(y_va, dtype=np.float64)
    model = MLP(X_tr.shape[1], hidden, weight_decay, seed)
    opt = Adam(model.params, lr=lr)
    rng = np.random.default_rng(seed+1)
    best_val, best_params, best_epoch, wait = np.inf, [p.copy() for p in model.params], 0, 0
    hist = {"train_loss": [], "val_loss": []}
    for epoch in range(1, max_epochs+1):
        order = rng.permutation(len(X_tr))
        for start in range(0, len(order), batch_size):
            idx = order[start:start+batch_size]
            p, cache = model.forward(X_tr[idx])
            model.set_params(opt.step(model.params, model.backward(p, y_tr[idx], cache)))
        tr_loss = bce(model.predict_proba(X_tr), y_tr)
        va_loss = bce(model.predict_proba(X_va), y_va)
        hist["train_loss"].append(tr_loss); hist["val_loss"].append(va_loss)
        if va_loss < best_val-1e-5:
            best_val, best_params, best_epoch, wait = va_loss, [p.copy() for p in model.params], epoch, 0
        else:
            wait += 1
            if wait >= patience: break
    model.set_params(best_params)
    return model, hist, best_epoch
