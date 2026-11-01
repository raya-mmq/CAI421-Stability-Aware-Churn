"""
CAI 421 - Checkpoint 2 - B1 Logistic Regression vs MLP baseline comparison (Student 4)

    python train_logistic_regression.py   # first
    python compare_models.py              # writes results/comparison/ and figures/logreg/

Inputs (read-only):
  results/mlp_results.json                    Student 5's saved MLP results (seed 42 = reported model)
  results/mlp_test_predictions_seed42.csv     Student 5's saved seed-42 test probabilities
  results/logreg/logreg_results.json          B1 results
  results/logreg/predictions/logreg_test_predictions.csv

The MLP is NOT retrained here. Its saved probabilities are re-scored with the same metric
function (train_mlp.metrics) and checked against the saved JSON before anything is compared.
Paired, class-stratified bootstrap on the fixed test set gives 95% percentile intervals for the
difference (LR - MLP). This quantifies test-sample uncertainty only; it does not cover variation
from a different data split.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss, precision_recall_curve,
                             roc_auc_score, roc_curve)

from preprocess import OUT_DIR as PROCESSED_DIR, load_splits
from train_mlp import metrics as shared_metrics

ROOT = Path(__file__).resolve().parent
CONFIG = json.load(open(ROOT / "configs" / "experiment_config.json"))
BOOT = CONFIG["logistic_regression"]["bootstrap"]
THRESHOLD = CONFIG["logistic_regression"]["threshold"]
OUT = ROOT / "results" / "comparison"
FIG = ROOT / "figures" / "logreg"
SCALAR = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier"]
LABELS = {"accuracy": "Accuracy", "precision": "Precision (churn)", "recall": "Recall (churn)",
          "f1": "F1 (churn)", "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC (average precision)",
          "brier": "Brier score (lower is better)"}


def main():
    OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    *_, y_te = load_splits()
    mlp_json = json.load(open(ROOT / "results" / "mlp_results.json"))
    lr_json = json.load(open(ROOT / "results" / "logreg" / "logreg_results.json"))
    mlp_pred = pd.read_csv(ROOT / "results" / "mlp_test_predictions_seed42.csv")
    lr_pred = pd.read_csv(ROOT / "results" / "logreg" / "predictions" / "logreg_test_predictions.csv")

    # ---- consistency checks: same test rows, same labels, saved metrics re-derivable ----
    idx_te = pd.read_csv(PROCESSED_DIR / "idx_test.csv")["raw_row_index"].values
    assert (lr_pred.raw_row_index.values == idx_te).all(), "LR predictions not in shared test order"
    assert (mlp_pred.y_true.values == y_te).all() and (lr_pred.y_true.values == y_te).all()
    p_mlp, p_lr = mlp_pred.p_churn.values, lr_pred.p_churn.values
    m_mlp, m_lr = shared_metrics(y_te, p_mlp, THRESHOLD), shared_metrics(y_te, p_lr, THRESHOLD)
    for k in SCALAR:
        assert np.isclose(m_mlp[k], mlp_json["test_seed42"][k], atol=1e-12), k
        assert np.isclose(m_lr[k], lr_json["test"][k], atol=1e-12), k
    assert m_mlp["confusion_matrix"] == mlp_json["test_seed42"]["confusion_matrix"]
    assert mlp_json["threshold"] == lr_json["threshold"] == THRESHOLD

    # ---- paired stratified bootstrap of LR - MLP ----
    rng = np.random.default_rng(BOOT["seed"])
    pos, neg = np.where(y_te == 1)[0], np.where(y_te == 0)[0]
    diffs = {k: [] for k in SCALAR}
    for _ in range(BOOT["n_resamples"]):
        i = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        a, b = shared_metrics(y_te[i], p_lr[i], THRESHOLD), shared_metrics(y_te[i], p_mlp[i], THRESHOLD)
        for k in SCALAR:
            diffs[k].append(a[k] - b[k])

    rows = []
    for k in SCALAR:
        d = np.array(diffs[k])
        lo, hi = np.percentile(d, [2.5, 97.5])
        rows.append({"metric": LABELS[k], "key": k,
                     "logistic_regression_B1": m_lr[k], "mlp_seed42": m_mlp[k],
                     "difference_lr_minus_mlp": m_lr[k] - m_mlp[k],
                     "bootstrap_ci95_low": lo, "bootstrap_ci95_high": hi,
                     "ci_excludes_zero": bool(lo > 0 or hi < 0),
                     "mlp_5seed_mean": mlp_json["test_5seed_mean_std"][k]["mean"],
                     "mlp_5seed_std": mlp_json["test_5seed_mean_std"][k]["std"]})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "lr_vs_mlp_test.csv", index=False)

    cm = {"logistic_regression_B1": m_lr["confusion_matrix"], "mlp_seed42": m_mlp["confusion_matrix"]}
    n_disagree = int(((p_lr >= THRESHOLD) != (p_mlp >= THRESHOLD)).sum())
    summary = {
        "test_n": int(len(y_te)), "test_n_churn": int(y_te.sum()), "threshold": THRESHOLD,
        "mlp_source": "results/mlp_results.json -> test_seed42 (saved by Student 5; not retrained)",
        "pr_auc_definition": CONFIG["logistic_regression"]["pr_auc_definition"],
        "bootstrap": BOOT,
        "confusion_matrices": cm,
        "n_test_customers_with_different_0.5_decision": n_disagree,
        "pearson_r_of_probabilities": float(np.corrcoef(p_lr, p_mlp)[0, 1]),
        "metrics": table.drop(columns="metric").set_index("key").to_dict(orient="index"),
        "note": "One fixed test split. Bootstrap intervals reflect test-sample uncertainty only.",
    }
    json.dump(summary, open(OUT / "lr_vs_mlp_test.json", "w"), indent=2)
    with pd.option_context("display.width", 200, "display.precision", 4):
        print(table.drop(columns="key").to_string(index=False))
    print("decisions that differ at 0.5:", n_disagree, " r(p_LR, p_MLP) =", summary["pearson_r_of_probabilities"])

    figure(y_te, p_lr, p_mlp, m_lr, m_mlp)


def figure(y, p_lr, p_mlp, m_lr, m_mlp):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BLUE, ORANGE, INK, MUTED, GRID_C = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 300})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 2.8))
    for p, m, c, ls, name in [(p_mlp, m_mlp, BLUE, "-", "MLP 45-32-1 (seed 42)"),
                              (p_lr, m_lr, ORANGE, "--", "Logistic regression B1")]:
        fpr, tpr, _ = roc_curve(y, p)
        a1.plot(fpr, tpr, color=c, ls=ls, lw=1.8, label=f"{name}: AUC {m['roc_auc']:.3f}")
        prec, rec, _ = precision_recall_curve(y, p)
        a2.step(rec, prec, where="post", color=c, ls=ls, lw=1.8, label=f"{name}: AP {m['pr_auc']:.3f}")
    a1.plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls=":", label="Chance")
    a2.axhline(y.mean(), color=MUTED, lw=0.8, ls=":", label=f"Prevalence {y.mean():.3f}")
    a1.set(xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1.01))
    a2.set(xlabel="Recall (churn)", ylabel="Precision (churn)", xlim=(0, 1), ylim=(0, 1.01))
    for ax, tag in [(a1, "(a) ROC, test n = 1,057"), (a2, "(b) Precision-recall, test")]:
        ax.grid(color=GRID_C, lw=0.6); ax.set_title(tag, fontsize=9, loc="left", color=INK)
        ax.legend(frameon=False, fontsize=6.5, loc="lower right" if ax is a1 else "upper right")
    fig.tight_layout(w_pad=2); fig.savefig(FIG / "lr_vs_mlp_roc_pr.png"); plt.close(fig)


if __name__ == "__main__":
    main()
