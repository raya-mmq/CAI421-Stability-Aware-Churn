"""Kernel SHAP (Lundberg & Lee, 2017) in plain NumPy for the team's NumPy MLP.

Value function: interventional, v(S) = mean_b f(x_S, background_b,~S) over a fixed
background set (the same definition shap.KernelExplainer uses).

Coalition design (same idea as shap.KernelExplainer, but with a larger budget):
  * players are the encoded columns whose value differs from at least one background row
    (columns that never vary get SHAP = 0 exactly, as in shap);
  * all coalitions of size 1..`enumerate_upto` and their complements are enumerated with
    their exact Shapley-kernel weights;
  * the remaining kernel weight is covered by `n_pairs` sampled coalitions (plus their
    complements), drawn with probability proportional to the kernel weight of their size;
  * the weighted least-squares problem is solved with the efficiency constraint
    sum(phi) = f(x) - E[f(background)] imposed exactly (no l1 regularisation).

Run `python kernel_shap_numpy.py` to execute the self-tests (exact match with brute-force
Shapley values on small problems).
"""
from __future__ import annotations
import itertools
import math
import numpy as np


def _kernel_total(M: int, s: int) -> float:
    """Total Shapley-kernel weight of all coalitions of size s (unnormalised)."""
    return (M - 1) / (s * (M - s))


def coalition_design(M: int, enumerate_upto: int, n_pairs: int, rng: np.random.Generator):
    """Return (Z, w): binary coalition matrix (K x M) and regression weights (K,)."""
    masks, weights = [], []
    enumerated = set()
    for s in range(1, M // 2 + 1):
        if s > enumerate_upto:
            break
        sizes = (s,) if s == M - s else (s, M - s)
        for size in sizes:
            per = _kernel_total(M, size) / math.comb(M, size)
            for c in itertools.combinations(range(M), size):
                z = np.zeros(M)
                z[list(c)] = 1.0
                masks.append(z)
                weights.append(per)
            enumerated.add(size)
    rest = [s for s in range(1, M) if s not in enumerated]
    if rest and n_pairs > 0:
        lower = sorted({min(s, M - s) for s in rest})
        p = np.array([_kernel_total(M, s) * (1 if s == M - s else 2) for s in lower])
        w_rest = float(sum(_kernel_total(M, s) for s in rest))
        p = p / p.sum()
        per = w_rest / (2 * n_pairs)
        for s in rng.choice(lower, size=n_pairs, p=p):
            z = np.zeros(M)
            z[rng.choice(M, size=s, replace=False)] = 1.0
            masks.extend([z, 1.0 - z])
            weights.extend([per, per])
    elif rest:
        raise ValueError("n_pairs must be > 0 when not every coalition size is enumerated")
    return np.asarray(masks), np.asarray(weights)


class KernelSHAP:
    def __init__(self, predict, background, enumerate_upto=2, n_pairs=4000, seed=0, chunk=1024,
                 first_layer=None, head=None):
        """first_layer=(W, b) and head (pre-activation -> output) are an optional speed-up for
        networks whose first layer is affine: X_mix @ W is then built from background @ W plus
        the coalition offsets instead of materialising every mixed input row. Results are
        identical to calling predict on the mixed rows."""
        self.predict = predict
        self.first_layer, self.head = first_layer, head
        self.background = np.asarray(background, dtype=np.float64)
        self.expected_value = float(np.mean(predict(self.background)))
        self.enumerate_upto, self.n_pairs, self.chunk = enumerate_upto, n_pairs, chunk
        self.rng = np.random.default_rng(seed)
        self._designs: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def _design(self, M):
        if M not in self._designs:  # one fixed design per player count, reused across customers
            self._designs[M] = coalition_design(M, self.enumerate_upto, self.n_pairs, self.rng)
        return self._designs[M]

    def explain_one(self, x):
        x = np.asarray(x, dtype=np.float64)
        bg = self.background
        n_feat = x.shape[0]
        phi = np.zeros(n_feat)
        fx = float(self.predict(x[None, :])[0])
        varying = np.where(~np.all(np.isclose(bg, x[None, :]), axis=0))[0]
        M = len(varying)
        if M == 0:
            return phi, fx
        if M == 1:
            phi[varying[0]] = fx - self.expected_value
            return phi, fx
        Z, w = self._design(M)
        delta = (x[None, :] - bg)[:, varying]                      # (B, M)
        v = np.empty(len(Z))
        if self.first_layer is not None:
            W, b = self.first_layer
            base = bg @ W + b                                        # (B, H)
            dW = delta[:, :, None] * W[varying][None, :, :]          # (B, M, H)
        for st in range(0, len(Z), self.chunk):
            Zc = Z[st:st + self.chunk]                               # (k, M)
            if self.first_layer is not None:
                pre = base[None, :, :] + np.einsum("km,bmh->kbh", Zc, dW, optimize=True)
                out = self.head(pre.reshape(-1, pre.shape[-1])).reshape(len(Zc), len(bg))
            else:
                X_mix = np.repeat(bg[None, :, :], len(Zc), axis=0)   # (k, B, F)
                X_mix[:, :, varying] += Zc[:, None, :] * delta[None, :, :]
                out = self.predict(X_mix.reshape(-1, n_feat)).reshape(len(Zc), len(bg))
            v[st:st + self.chunk] = out.mean(axis=1)
        total = fx - self.expected_value
        # eliminate the last player using the efficiency constraint, then weighted least squares
        y = v - self.expected_value - Z[:, -1] * total
        A = Z[:, :-1] - Z[:, [-1]]
        Aw = A * w[:, None]
        sol = np.linalg.solve(Aw.T @ A, Aw.T @ y)
        phi[varying[:-1]] = sol
        phi[varying[-1]] = total - sol.sum()
        return phi, fx

    def shap_values(self, X):
        X = np.asarray(X, dtype=np.float64)
        return np.stack([self.explain_one(x)[0] for x in X])


# ----------------------------------------------------------------------------- self-tests
def _brute_force_shapley(predict, x, bg):
    M = len(x)
    def v(S):
        Xm = bg.copy()
        Xm[:, list(S)] = x[list(S)]
        return predict(Xm).mean()
    phi = np.zeros(M)
    for j in range(M):
        others = [k for k in range(M) if k != j]
        for r in range(M):
            for S in itertools.combinations(others, r):
                wgt = math.factorial(r) * math.factorial(M - r - 1) / math.factorial(M)
                phi[j] += wgt * (v(S + (j,)) - v(S))
    return phi


def _self_test():
    rng = np.random.default_rng(1)
    W1, b1 = rng.normal(size=(8, 6)), rng.normal(size=6)
    W2, b2 = rng.normal(size=(6, 1)), rng.normal(size=1)
    f = lambda X: 1 / (1 + np.exp(-(np.maximum(X @ W1 + b1, 0) @ W2 + b2).ravel()))
    bg = rng.normal(size=(7, 8))
    x = rng.normal(size=8)
    exact = _brute_force_shapley(f, x, bg)
    full = KernelSHAP(f, bg, enumerate_upto=4, n_pairs=0).explain_one(x)[0]
    assert np.allclose(full, exact, atol=1e-10), (full, exact)
    sampled = KernelSHAP(f, bg, enumerate_upto=1, n_pairs=20000, seed=3).explain_one(x)[0]
    assert np.abs(sampled - exact).max() < 0.02, np.abs(sampled - exact).max()
    assert abs(sampled.sum() - (f(x[None])[0] - f(bg).mean())) < 1e-12
    head = lambda H: 1 / (1 + np.exp(-(np.maximum(H, 0) @ W2 + b2).ravel()))
    fast = KernelSHAP(f, bg, enumerate_upto=1, n_pairs=20000, seed=3, first_layer=(W1, b1), head=head).explain_one(x)[0]
    assert np.allclose(fast, sampled, atol=1e-10), "fast path must equal the generic path"
    # linear model: Shapley value = w_j * (x_j - mean background_j)
    w = rng.normal(size=8)
    lin = lambda X: X @ w
    est = KernelSHAP(lin, bg, enumerate_upto=1, n_pairs=200, seed=4).explain_one(x)[0]
    assert np.allclose(est, w * (x - bg.mean(0)), atol=1e-10)
    print("kernel_shap_numpy self-tests passed: exact match with brute-force Shapley (M=8), "
          f"sampled max error {np.abs(sampled - exact).max():.4f}, linear-model check exact.")


if __name__ == "__main__":
    _self_test()
