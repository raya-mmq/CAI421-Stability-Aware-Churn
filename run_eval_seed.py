import sys, time
import numpy as np
from pathlib import Path
import run_shap_stability as R
seed = int(sys.argv[1])
split = 'test'
out = R.ROOT / 'results' / 'shap_eval'
out.mkdir(parents=True, exist_ok=True)
X, y_train, y_val, names, groups = R.load_data()
bg = X['train'][np.random.default_rng(R.BACKGROUND_SEED).choice(len(X['train']), R.BACKGROUND_SIZE, replace=False)]
t0 = time.time()
model, hist, best = R.train(X['train'], y_train, X['val'], y_val, hidden=R.HIDDEN, seed=seed, **R.TRAINING)
expected, fn = R.make_explainer(model, bg, 'numpy', R.SHAP_SEED[split], 'auto')
enc = fn(X[split])
agg, orig = R.aggregate_to_original(enc, names, groups)
p = model.predict_proba(X[split])
np.save(out / f'aggregated_shap_{split}_seed_{seed}.npy', agg)
np.save(out / f'probs_{split}_seed_{seed}.npy', p)
print(f'seed {seed} done in {time.time() - t0:.0f}s, best epoch {best}, max additivity err {np.max(np.abs(p - (expected + enc.sum(1)))):.1e}')
