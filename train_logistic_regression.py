"""
CAI 421 - Checkpoint 2 - Logistic Regression baseline B1 (Student 4)

    python preprocess.py                 # once (Student 3's shared pipeline, unchanged)
    python train_logistic_regression.py  # writes results/logreg/ and figures/logreg/

Protocol (no test-set leakage):
  1. Load the SHARED processed splits with preprocess.load_splits() - the exact 45 features,
     feature order and train/val/test rows used by the MLP. Nothing is re-split, re-scaled,
     resampled (no SMOTE) or feature-selected here.
  2. Verify the processed files against the SHA-256 values in configs/experiment_config.json.
  3. Fit L2 logistic regression on TRAIN for each C in a small pre-declared grid.
  4. Select C by VALIDATION ROC-AUC (the same selection metric the MLP used); ties -> smaller C.
  5. Evaluate the frozen model ONCE on TEST at threshold 0.5.
Metric definitions are imported from train_mlp.metrics (Student 5) so both models are scored
by identical code; PR-AUC = sklearn average_precision_score, exactly as for the MLP.

Regularisation is L2 (scikit-learn's default; the `penalty` argument is deprecated from 1.8, so it is
not passed). lbfgs is deterministic; random_state=42 is still set for completeness.
"""
import hashlib
import json
import platform
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from preprocess import OUT_DIR as PROCESSED_DIR, load_splits   # Student 3 (unchanged)
from train_mlp import metrics as shared_metrics                # Student 5 (unchanged)

ROOT = Path(__file__).resolve().parent
CONFIG = json.load(open(ROOT / "configs" / "experiment_config.json"))
CFG = CONFIG["logistic_regression"]
RES = ROOT / "results" / "logreg"
PRED = RES / "predictions"          # per-customer outputs: git-ignored, regenerate locally
FIG = ROOT / "figures" / "logreg"
SEED = CFG["seed"]
THRESHOLD = CFG["threshold"]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_processed_data():
    """Fail loudly if the processed files are not the team's reference files."""
    expected = CONFIG["data_integrity"]["processed_sha256"]
    bad = {f: sha256(PROCESSED_DIR / f) for f in expected if sha256(PROCESSED_DIR / f) != expected[f]}
    if bad:
        sys.exit(f"Processed data differ from the team reference: {sorted(bad)}.\n"
                 "Re-run `python preprocess.py` on the reference CSV (see DATA_README.md).")
    return {"all_processed_files_match_reference": True, "n_files_checked": len(expected)}


def to_jsonable(m):
    return {k: (v if isinstance(v, dict) else float(v)) for k, v in m.items()}


def main():
    np.random.seed(SEED)
    RES.mkdir(parents=True, exist_ok=True); PRED.mkdir(exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)

    integrity = verify_processed_data()
    X_tr, X_va, X_te, y_tr, y_va, y_te = load_splits()
    # load_splits() returns float32; cast to float64 exactly as the MLP does (mlp.train / predict_proba)
    X_tr, X_va, X_te = (np.asarray(X, dtype=np.float64) for X in (X_tr, X_va, X_te))
    feature_names = json.load(open(PROCESSED_DIR / "feature_names.json"))
    idx_te = pd.read_csv(PROCESSED_DIR / "idx_test.csv")["raw_row_index"].values
    sc = CONFIG["shared_pipeline"]
    assert X_tr.shape == (sc["n_train"], sc["n_features"]), X_tr.shape
    assert X_va.shape == (sc["n_val"], sc["n_features"]), X_va.shape
    assert X_te.shape == (sc["n_test"], sc["n_features"]), X_te.shape
    assert len(feature_names) == sc["n_features"]

    # ---------------- 1. validation-only selection of C ----------------
    rows, models = [], {}
    for C in CFG["grid"]["C"]:
        model = LogisticRegression(C=C, **CFG["fixed_params"]).fit(X_tr, y_tr)
        p_va = model.predict_proba(X_va)[:, 1]
        m_va = shared_metrics(y_va, p_va, THRESHOLD)
        rows.append({"C": C, "n_iter": int(model.n_iter_[0]),
                     "converged": bool(model.n_iter_[0] < CFG["fixed_params"]["max_iter"]),
                     **{f"val_{k}": v for k, v in m_va.items() if k != "confusion_matrix"}})
        models[C] = model
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[-1].items()})
    grid = pd.DataFrame(rows)
    grid.to_csv(RES / "logreg_validation_grid.csv", index=False)
    best_auc = grid["val_roc_auc"].max()
    best_C = float(grid.loc[np.isclose(grid["val_roc_auc"], best_auc, rtol=0, atol=1e-12), "C"].min())
    model = models[best_C]
    print(f"Selected on validation: C = {best_C}")

    # ---------------- 2. frozen model: validation + single test evaluation ----------------
    p_tr = model.predict_proba(X_tr)[:, 1]
    p_va = model.predict_proba(X_va)[:, 1]
    p_te = model.predict_proba(X_te)[:, 1]
    res_tr, res_va, res_te = (shared_metrics(y, p, THRESHOLD) for y, p in
                              [(y_tr, p_tr), (y_va, p_va), (y_te, p_te)])

    # default-C reference row (C = 1.0), reported for transparency only, not selected on test
    p_te_default = models[1.0].predict_proba(X_te)[:, 1] if 1.0 in models else None

    joblib.dump(model, RES / "logreg_model.joblib")
    pd.DataFrame({"feature": feature_names, "coefficient": model.coef_[0],
                  "odds_ratio_per_unit": np.exp(model.coef_[0])}) \
        .assign(abs_coef=lambda d: d.coefficient.abs()).sort_values("abs_coef", ascending=False) \
        .drop(columns="abs_coef").to_csv(RES / "logreg_coefficients.csv", index=False)
    pd.DataFrame({"raw_row_index": idx_te, "y_true": y_te, "p_churn": p_te,
                  "y_pred": (p_te >= THRESHOLD).astype(int)}) \
        .to_csv(PRED / "logreg_test_predictions.csv", index=False)
    pd.DataFrame({"raw_row_index": pd.read_csv(PROCESSED_DIR / "idx_val.csv")["raw_row_index"].values,
                  "y_true": y_va, "p_churn": p_va, "y_pred": (p_va >= THRESHOLD).astype(int)}) \
        .to_csv(PRED / "logreg_val_predictions.csv", index=False)

    out = {
        "model": "LogisticRegression (scikit-learn)",
        "selected_params": {"C": best_C, "penalty": "l2 (scikit-learn default)", **CFG["fixed_params"]},
        "input_dtype": "float64",
        "intercept": float(model.intercept_[0]),
        "n_parameters": int(model.coef_.size + model.intercept_.size),
        "n_iter": int(model.n_iter_[0]),
        "selection": {"split": "validation", "metric": "ROC-AUC", "grid_C": CFG["grid"]["C"],
                      "rule": CFG["selection_rule"]},
        "threshold": THRESHOLD,
        "pr_auc_definition": CFG["pr_auc_definition"],
        "metric_function": "train_mlp.metrics (identical to the MLP)",
        "n": {"train": int(len(y_tr)), "val": int(len(y_va)), "test": int(len(y_te))},
        "n_churn": {"train": int(y_tr.sum()), "val": int(y_va.sum()), "test": int(y_te.sum())},
        "n_features": int(X_tr.shape[1]),
        "train": to_jsonable(res_tr),
        "val": to_jsonable(res_va),
        "test": to_jsonable(res_te),
        "test_default_C1_reference": (to_jsonable(shared_metrics(y_te, p_te_default, THRESHOLD))
                                      if p_te_default is not None else None),
        "data_integrity": integrity,
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
                        "platform": f"{platform.machine()} {platform.system()}"},
    }
    json.dump(out, open(RES / "logreg_results.json", "w"), indent=2)
    print(json.dumps({"selected_C": best_C, "val": out["val"], "test": out["test"],
                      "n_test": out["n"]["test"]}, indent=2))

    figure_confusion(res_te["confusion_matrix"])


def figure_confusion(cm):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, MUTED = "#0b0b0b", "#52514e"
    plt.rcParams.update({"font.size": 9, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "figure.dpi": 300})
    M = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    fig, ax = plt.subplots(figsize=(2.8, 2.5))
    ax.imshow(M, cmap="Oranges", vmin=0)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{M[i, j]}", ha="center", va="center", fontsize=11,
                    color="white" if M[i, j] > M.max() / 2 else INK)
    ax.set_xticks([0, 1], ["No churn", "Churn"]); ax.set_yticks([0, 1], ["No churn", "Churn"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout(); fig.savefig(FIG / "lr_confusion_matrix.png"); plt.close(fig)


if __name__ == "__main__":
    main()
