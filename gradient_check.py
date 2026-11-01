"""
Numerical check of the backpropagation formulas used in mlp.py (and proved by
induction in the Method section). Compares the analytic gradient with a
central finite difference on a small batch of the real training data.

    python gradient_check.py
"""
import json
from pathlib import Path

import numpy as np

from mlp import MLP
from preprocess import load_splits

ROOT = Path(__file__).resolve().parent


def main():
    X_train, _, _, y_train, _, _ = load_splits()
    X = X_train[:32].astype(np.float64)
    y = y_train[:32].astype(np.float64)

    model = MLP(X.shape[1], hidden=(16, 8, 4), weight_decay=1e-3, seed=0)   # 3 hidden layers
    y_hat, cache = model.forward(X)
    analytic = model.backward(y_hat, y, cache)

    h = 1e-6
    max_rel = 0.0
    params = model.params
    for i, p in enumerate(params):
        num = np.zeros_like(p)
        for j in np.ndindex(p.shape):
            plus = [q.copy() for q in params]; plus[i][j] += h
            minus = [q.copy() for q in params]; minus[i][j] -= h
            model.set_params(plus); lp = model.loss(X, y)
            model.set_params(minus); lm = model.loss(X, y)
            num[j] = (lp - lm) / (2 * h)
        model.set_params(params)
        rel = np.linalg.norm(num - analytic[i]) / max(np.linalg.norm(num) + np.linalg.norm(analytic[i]), 1e-12)
        max_rel = max(max_rel, rel)

    out = {"layers": model.sizes, "n_params": int(sum(p.size for p in params)),
           "max_relative_error": float(max_rel), "passed": bool(max_rel < 1e-6)}
    (ROOT / "results").mkdir(exist_ok=True)
    json.dump(out, open(ROOT / "results" / "gradient_check.json", "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
