"""CAI 421 - Student 2: five-seed SHAP explanation stability for the NumPy MLP.

Run once per split (they can run in parallel):
    python run_shap_stability.py --split val
    python run_shap_stability.py --split test

Validation outputs feed threshold selection (abstention gate); test outputs are used
once, for the held-out evaluation. Test labels are never read by this script.

SHAP backend
------------
Default backend "numpy" = Kernel SHAP from kernel_shap_numpy.py: same interventional
value function as shap.KernelExplainer, but every coalition of size 1, 2, M-2 and M-1
is enumerated exactly and 2,000 further complementary pairs are sampled (about 6,000
coalitions per customer). The earlier run used shap.KernelExplainer with only 128
coalitions per customer for up to 45 players, which left enough Monte-Carlo noise to
change a model's own top-3 feature set for roughly 1 customer in 5 (see README).
`--backend shap` is kept for anyone who wants to cross-check with the shap package.
"""
from __future__ import annotations

import argparse
import json
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from kernel_shap_numpy import KernelSHAP
from mlp import sigmoid, train

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "processed"
DEFAULT_OUT = ROOT / "results" / "shap_stability"

GATE_SEEDS = [1, 2, 3, 4, 5]          # disjoint from baseline seeds 42-46 and evaluation seeds 101-105
HIDDEN = (32,)
TRAINING = dict(lr=0.003, batch_size=128, weight_decay=1e-4, max_epochs=300, patience=20)
BACKGROUND_SEED, BACKGROUND_SIZE = 2026, 50
SHAP_SEED = {"val": 2718, "test": 2719}
ENUMERATE_UPTO, N_PAIRS = 2, 2000      # numpy backend budget
NOISE_SEED_OFFSET = 1000               # second, independent coalition design for the noise check
TOP_K = 3


# ----------------------------------------------------------------------------- data
def load_data():
    frames = {s: pd.read_csv(DATA / f"X_{s}.csv") for s in ("train", "val", "test")}
    names = json.loads((DATA / "feature_names.json").read_text())
    groups = json.loads((DATA / "feature_groups.json").read_text())
    for split, frame in frames.items():
        if list(frame.columns) != names:
            raise ValueError(f"{split} feature order differs from feature_names.json")
    if len(names) != 45 or len(groups) != 19:
        raise ValueError(f"Expected 45 encoded and 19 original features; got {len(names)} and {len(groups)}")
    encoded_in_groups = [c for cols in groups.values() for c in cols]
    if sorted(encoded_in_groups) != sorted(names):
        raise ValueError("feature_groups.json must cover every encoded column exactly once")
    y_train = pd.read_csv(DATA / "y_train.csv").iloc[:, 0].to_numpy(dtype=int)
    y_val = pd.read_csv(DATA / "y_val.csv").iloc[:, 0].to_numpy(dtype=int)
    X = {k: v.to_numpy(dtype=np.float64) for k, v in frames.items()}
    return X, y_train, y_val, names, groups


# ----------------------------------------------------------------------------- SHAP
def _normalize_shap_output(values, n_rows, n_features):
    """shap returns list / 2-D / 3-D arrays depending on version; return (n_rows, n_features)."""
    if isinstance(values, list):
        values = values[-1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, -1] if values.shape[:2] == (n_rows, n_features) else values[-1]
    if values.shape != (n_rows, n_features):
        raise ValueError(f"Expected SHAP shape {(n_rows, n_features)}, got {values.shape}")
    return values


def make_explainer(model, background, backend, seed, nsamples):
    """Return (expected_value, shap_values(X) -> (n, 45) array)."""
    if backend == "numpy":
        def head(H):  # first-layer pre-activation -> churn probability
            a = np.maximum(H, 0.0)
            for W, b in zip(model.W[1:-1], model.b[1:-1]):
                a = np.maximum(a @ W + b, 0.0)
            return sigmoid(a @ model.W[-1] + model.b[-1]).ravel()

        ex = KernelSHAP(model.predict_proba, background, enumerate_upto=ENUMERATE_UPTO,
                        n_pairs=N_PAIRS, seed=seed, first_layer=(model.W[0], model.b[0]), head=head)
        return ex.expected_value, ex.shap_values

    import shap  # optional dependency, only for --backend shap
    ex = shap.KernelExplainer(model.predict_proba, background, link="identity")

    def values(X):
        np.random.seed(seed)  # same coalition stream for every gate model
        raw = ex.shap_values(X, nsamples=nsamples, l1_reg=0.0, silent=True)
        return _normalize_shap_output(raw, len(X), X.shape[1])

    return float(np.asarray(ex.expected_value).reshape(-1)[0]), values


def aggregate_to_original(encoded, feature_names, groups):
    """Sum signed one-hot SHAP values within each original feature."""
    col = {n: i for i, n in enumerate(feature_names)}
    originals = list(groups)
    out = np.stack([encoded[:, [col[c] for c in groups[g]]].sum(axis=1) for g in originals], axis=1)
    return out, originals


def top_k(aggregated, names, k=TOP_K):
    order = np.argsort(-np.abs(aggregated), axis=1, kind="stable")[:, :k]
    return [[names[i] for i in row] for row in order]


def jaccard_rows(tops_a, tops_b):
    return np.array([len(set(a) & set(b)) / len(set(a) | set(b)) for a, b in zip(tops_a, tops_b)])


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--seed-list", type=int, nargs="+", default=GATE_SEEDS,
                    help="gate seeds (default 1-5); use 101 102 103 104 105 with --out-dir for evaluation models")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--background-size", type=int, default=BACKGROUND_SIZE)
    ap.add_argument("--backend", choices=["numpy", "shap"], default="numpy")
    ap.add_argument("--nsamples", default="auto", help="shap backend only (int or 'auto')")
    ap.add_argument("--noise-check", type=int, default=200,
                    help="customers re-explained with a second coalition design for the first seed (0 = skip)")
    ap.add_argument("--max-samples", type=int, default=0, help="debug only; 0 = complete split")
    args = ap.parse_args()
    if len(args.seed_list) < 2:
        ap.error("at least two seeds are required")
    nsamples = args.nsamples if args.nsamples == "auto" else int(args.nsamples)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    X, y_train, y_val, feature_names, groups = load_data()
    split = args.split
    X_explain = X[split][: args.max_samples] if args.max_samples else X[split]
    n = len(X_explain)
    shap_seed = SHAP_SEED[split]

    bg_idx = np.random.default_rng(BACKGROUND_SEED).choice(len(X["train"]), args.background_size, replace=False)
    background = X["train"][bg_idx]
    np.save(out_dir / "background_indices.npy", bg_idx)
    pd.DataFrame({"background_train_row_position": bg_idx}).to_csv(out_dir / "background_indices.csv", index=False)
    (out_dir / "aggregated_feature_order.json").write_text(json.dumps(list(groups), indent=2))

    tops, probs, agg_all, model_rows, additivity, noise = {}, {}, [], [], [], None
    t0 = time.time()
    for seed in args.seed_list:
        print(f"[SHAP] {split}: training gate MLP seed={seed}", flush=True)
        model, hist, best_epoch = train(X["train"], y_train, X["val"], y_val, hidden=HIDDEN, seed=seed, **TRAINING)
        np.savez(out_dir / f"gate_mlp_seed_{seed}_weights.npz",
                 **{f"W{i + 1}": w for i, w in enumerate(model.W)}, **{f"b{i + 1}": b for i, b in enumerate(model.b)})
        pd.DataFrame({"epoch": np.arange(1, len(hist["train_loss"]) + 1), "train_bce": hist["train_loss"],
                      "validation_bce": hist["val_loss"]}).to_csv(out_dir / f"training_history_seed_{seed}.csv", index=False)

        expected, shap_fn = make_explainer(model, background, args.backend, shap_seed, nsamples)
        encoded = shap_fn(X_explain)
        aggregated, original_names = aggregate_to_original(encoded, feature_names, groups)
        p = model.predict_proba(X_explain)
        err = p - (expected + encoded.sum(axis=1))
        additivity.append((float(np.mean(np.abs(err))), float(np.max(np.abs(err)))))

        tops[seed] = top_k(aggregated, original_names)
        probs[seed] = p
        agg_all.append(aggregated)
        np.save(out_dir / f"aggregated_shap_{split}_seed_{seed}.npy", aggregated)
        pd.DataFrame(aggregated, columns=original_names).to_csv(
            out_dir / f"aggregated_shap_{split}_seed_{seed}.csv", index=False, float_format="%.8g")
        pd.DataFrame(tops[seed], columns=[f"top_{i}" for i in range(1, TOP_K + 1)]).to_csv(
            out_dir / f"top3_features_{split}_seed_{seed}.csv", index=False)
        model_rows.append({"seed": seed, "best_epoch": best_epoch, "epochs_run": len(hist["val_loss"]),
                           "best_validation_bce": float(min(hist["val_loss"])),
                           "validation_roc_auc": float(roc_auc_score(y_val, model.predict_proba(X["val"]))),
                           f"mean_predicted_probability_{split}": float(p.mean())})

        # Monte-Carlo noise floor: same model, independent coalition design.
        if noise is None and args.noise_check:
            m = min(args.noise_check, n)
            _, shap_fn2 = make_explainer(model, background, args.backend, shap_seed + NOISE_SEED_OFFSET, nsamples)
            agg2, _ = aggregate_to_original(shap_fn2(X_explain[:m]), feature_names, groups)
            j = jaccard_rows(tops[seed][:m], top_k(agg2, original_names))
            noise = {"seed": seed, "n_customers": m, "mean_same_model_jaccard": float(j.mean()),
                     "share_identical_top3": float(np.mean(j == 1.0)),
                     "max_abs_shap_difference": float(np.max(np.abs(agg2 - aggregated[:m])))}
            print(f"[SHAP] noise check: same-model top-3 Jaccard {j.mean():.3f}", flush=True)
        print(f"[SHAP] {split}: seed={seed} done ({time.time() - t0:.0f}s); max additivity error {np.max(np.abs(err)):.1e}",
              flush=True)

    seeds = list(args.seed_list)
    pairs = list(combinations(seeds, 2))
    pair_scores = np.stack([jaccard_rows(tops[a], tops[b]) for a, b in pairs], axis=1)
    stability = pair_scores.mean(axis=1)

    raw_idx = pd.read_csv(DATA / f"idx_{split}.csv").iloc[:n, 0].to_numpy()
    out = pd.DataFrame({"row_position": np.arange(n), "raw_row_index": raw_idx, "shap_stability": stability,
                        "mean_gate_probability": np.mean([probs[s] for s in seeds], axis=0)})
    for s in seeds:
        out[f"seed_{s}_probability"] = probs[s]
    for s in seeds:
        for r in range(TOP_K):
            out[f"seed_{s}_top_{r + 1}"] = [t[r] for t in tops[s]]
    out.to_csv(out_dir / f"customer_shap_stability_{split}.csv", index=False, float_format="%.6f")
    pd.DataFrame(pair_scores, columns=[f"jaccard_{a}_{b}" for a, b in pairs]).to_csv(
        out_dir / f"pairwise_jaccard_{split}.csv", index=False, float_format="%.6f")

    mean_abs = np.mean(np.abs(np.stack(agg_all)), axis=(0, 1))
    pd.DataFrame({"original_feature": original_names, "mean_absolute_aggregated_shap": mean_abs}) \
        .sort_values("mean_absolute_aggregated_shap", ascending=False) \
        .to_csv(out_dir / f"global_mean_abs_shap_{split}.csv", index=False, float_format="%.6f")
    pd.DataFrame(model_rows).to_csv(out_dir / f"gate_model_training_summary_{split}.csv", index=False)

    top1_unanimous = np.mean([len({tops[s][i][0] for s in seeds}) == 1 for i in range(n)])
    stats = {"n_customers": n, "mean": float(stability.mean()), "std": float(stability.std(ddof=1)),
             "median": float(np.median(stability)), "q25": float(np.quantile(stability, 0.25)),
             "q75": float(np.quantile(stability, 0.75)), "min": float(stability.min()), "max": float(stability.max()),
             "share_below_0_5": float(np.mean(stability < 0.5)), "share_at_least_0_5": float(np.mean(stability >= 0.5)),
             "share_identical_top3_all_models": float(np.mean(stability == 1.0)),
             "share_same_top1_all_models": float(top1_unanimous)}
    summary = {
        "split": split,
        "method": ("Kernel SHAP (kernel_shap_numpy.py; exact enumeration of coalition sizes 1, 2, M-2, M-1 "
                   f"plus {N_PAIRS} sampled complementary pairs)" if args.backend == "numpy"
                   else f"shap.KernelExplainer, nsamples={nsamples}"),
        "backend": args.backend, "gate_seeds": seeds, "architecture": [45, *HIDDEN, 1],
        "training_parameters": TRAINING, "background_size": len(background), "background_seed": BACKGROUND_SEED,
        "shap_random_seed": shap_seed, "link": "identity (probability output)", "top_k": TOP_K,
        "encoded_features": len(feature_names), "original_features": len(groups),
        "aggregation": "sum signed one-hot SHAP values within each original feature; rank by absolute aggregated value",
        "stability": f"mean pairwise Jaccard overlap of top-{TOP_K} sets across {len(pairs)} gate-model pairs",
        "n_seed_pairs": len(pairs),
        "efficiency_check": {
                             "mean_abs_error": float(np.mean([a for a, _ in additivity])),
                             "max_abs_error": float(np.max([b for _, b in additivity]))},
        "monte_carlo_noise_check": noise, "stability_summary": stats,
        "runtime_seconds": round(time.time() - t0, 1), "full_split": not bool(args.max_samples),
    }
    (out_dir / f"shap_stability_summary_{split}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(stats, indent=2), flush=True)

    vp, tp = out_dir / "shap_stability_summary_val.json", out_dir / "shap_stability_summary_test.json"
    if vp.exists() and tp.exists():
        merged = {"validation": json.loads(vp.read_text()), "test": json.loads(tp.read_text())}
        (out_dir / "shap_stability_summary.json").write_text(json.dumps(merged, indent=2))


if __name__ == "__main__":
    main()
