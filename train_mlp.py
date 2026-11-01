"""
CAI 421 - Checkpoint 2 - MLP baseline: tuning, training and evaluation (Student 5)

    python preprocess.py        # once (Student 3's shared pipeline)
    python gradient_check.py    # optional: verifies backpropagation numerically
    python train_mlp.py         # writes results/ and figures/

Protocol (no test-set leakage):
  1. Grid search over architecture / learning rate / batch size / weight decay,
     each config trained on TRAIN with early stopping on VALIDATION BCE (seed 42).
  2. The config with the highest VALIDATION ROC-AUC is selected and frozen.
  3. The frozen config is evaluated ONCE on TEST (seed 42 = reported model),
     and re-trained with 5 seeds (42-46) to report mean +/- std.
Decision threshold for Accuracy / Precision / Recall / F1 is fixed at 0.5.
"""
import itertools
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)

from mlp import train
from preprocess import load_splits

ROOT = Path(__file__).resolve().parent
RES, FIG = ROOT / "results", ROOT / "figures"
SEED = 42
EVAL_SEEDS = [42, 43, 44, 45, 46]
THRESHOLD = 0.5

GRID = {
    "hidden": [(32,), (64, 32), (128, 64)],
    "lr": [1e-3, 3e-3],
    "batch_size": [64, 128],
    "weight_decay": [0.0, 1e-4],
}
FIXED = {"max_epochs": 300, "patience": 20}


def metrics(y, p, thr=THRESHOLD):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "brier": brier_score_loss(y, p),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def main():
    RES.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
    X_tr, X_va, X_te, y_tr, y_va, y_te = load_splits()

    # ---------------- 1. validation-only tuning ----------------
    rows = []
    keys = list(GRID)
    for values in itertools.product(*GRID.values()):
        cfg = dict(zip(keys, values))
        t0 = time.time()
        model, hist, best_ep = train(X_tr, y_tr, X_va, y_va, seed=SEED, **cfg, **FIXED)
        p_va = model.predict_proba(X_va)
        rows.append({**{k: str(v) for k, v in cfg.items()},
                     "best_epoch": best_ep, "epochs_run": len(hist["val_loss"]),
                     "val_bce": min(hist["val_loss"]), "val_roc_auc": roc_auc_score(y_va, p_va),
                     "seconds": round(time.time() - t0, 2)})
        print(rows[-1])
    tuning = pd.DataFrame(rows).sort_values("val_roc_auc", ascending=False)
    tuning.to_csv(RES / "mlp_tuning.csv", index=False)
    best_row = tuning.iloc[0]
    best_cfg = {"hidden": eval(best_row["hidden"]), "lr": float(best_row["lr"]),
                "batch_size": int(best_row["batch_size"]),
                "weight_decay": float(best_row["weight_decay"])}
    print("Selected on validation:", best_cfg)

    # ---------------- 2. frozen config, multiple seeds ----------------
    per_seed = {}
    for s in EVAL_SEEDS:
        t0 = time.time()
        model, hist, best_ep = train(X_tr, y_tr, X_va, y_va, seed=s, **best_cfg, **FIXED)
        secs = time.time() - t0
        p_va, p_te = model.predict_proba(X_va), model.predict_proba(X_te)
        per_seed[s] = {"best_epoch": best_ep, "epochs_run": len(hist["val_loss"]),
                       "train_seconds": round(secs, 2),
                       "val": metrics(y_va, p_va), "test": metrics(y_te, p_te)}
        if s == SEED:
            main_model, main_hist, main_p_te = model, hist, p_te
            np.savez(RES / "mlp_seed42_weights.npz",
                     **{f"W{i+1}": W for i, W in enumerate(model.W)},
                     **{f"b{i+1}": b for i, b in enumerate(model.b)})
            pd.DataFrame({"y_true": y_te, "p_churn": p_te}).to_csv(RES / "mlp_test_predictions_seed42.csv", index=False)

    scalar = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier"]
    summary = {m: {"mean": float(np.mean([per_seed[s]["test"][m] for s in EVAL_SEEDS])),
                   "std": float(np.std([per_seed[s]["test"][m] for s in EVAL_SEEDS], ddof=1))}
               for m in scalar}

    n_params = int(sum(W.size for W in main_model.W) + sum(b.size for b in main_model.b))
    out = {
        "selected_config": {**best_cfg, "hidden": list(best_cfg["hidden"]), **FIXED,
                            "selection_metric": "validation ROC-AUC",
                            "early_stopping": "validation BCE, restore best epoch"},
        "architecture": [X_tr.shape[1], *best_cfg["hidden"], 1],
        "n_parameters": n_params,
        "threshold": THRESHOLD,
        "reported_model_seed": SEED,
        "test_seed42": per_seed[SEED]["test"],
        "val_seed42": per_seed[SEED]["val"],
        "test_5seed_mean_std": summary,
        "per_seed": {str(k): v for k, v in per_seed.items()},
        "n_configs_tuned": len(tuning),
        "hardware": f"{platform.processor() or platform.machine()} CPU, {platform.system()}, Python {platform.python_version()}, NumPy {np.__version__}",
    }
    json.dump(out, open(RES / "mlp_results.json", "w"), indent=2, default=float)
    print(json.dumps({"config": out["selected_config"], "test_seed42": out["test_seed42"],
                      "test_5seed": summary}, indent=2, default=float))

    figures(main_hist, per_seed[SEED], y_te, main_p_te)


def figures(hist, res, y_te, p_te):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    BLUE, ORANGE, INK, MUTED, GRID_C = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 300})
    cm = res["test"]["confusion_matrix"]
    M = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    fpr, tpr, _ = roc_curve(y_te, p_te)

    def learning(ax, small=False):
        ep = np.arange(1, len(hist["train_loss"]) + 1)
        ax.plot(ep, hist["train_loss"], color=BLUE, lw=1.5, label="Train")
        ax.plot(ep, hist["val_loss"], color=ORANGE, lw=1.5, label="Validation")
        ax.axvline(res["best_epoch"], color=MUTED, lw=0.8, ls="--")
        ax.text(res["best_epoch"], ax.get_ylim()[1], f" best epoch {res['best_epoch']}",
                va="top", fontsize=7, color=MUTED)
        ax.set_xlabel("Epoch"); ax.set_ylabel("BCE loss")
        ax.grid(axis="y", color=GRID_C, lw=0.6)
        ax.legend(frameon=False, fontsize=7 if small else 8, loc="lower left" if small else "best")

    def confusion(ax, small=False):
        ax.imshow(M, cmap="Blues", vmin=0, aspect="auto" if small else "equal")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{M[i, j]}", ha="center", va="center", fontsize=11,
                        color="white" if M[i, j] > M.max() / 2 else INK)
        ax.set_xticks([0, 1], ["No churn", "Churn"]); ax.set_yticks([0, 1], ["No churn", "Churn"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        for s in ax.spines.values(): s.set_visible(False)

    def roc(ax, small=False):
        ax.plot(fpr, tpr, color=BLUE, lw=1.8, label=f"MLP (AUC = {res['test']['roc_auc']:.3f})")
        ax.plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls="--", label="Chance")
        ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.01); ax.grid(color=GRID_C, lw=0.6)
        ax.legend(frameon=False, fontsize=7 if small else 8, loc="lower right", handlelength=1.5 if small else 2)

    # individual figures
    for name, draw, size in [("mlp_learning_curve", learning, (3.4, 2.4)),
                             ("mlp_confusion_matrix", confusion, (2.8, 2.5)),
                             ("mlp_roc_curve", roc, (3.0, 2.8))]:
        fig, ax = plt.subplots(figsize=size)
        draw(ax)
        fig.tight_layout(); fig.savefig(FIG / f"{name}.png"); plt.close(fig)

    # combined three-panel figure for the paper
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4), gridspec_kw={"width_ratios": [1.25, 0.9, 1.0]})
    for ax, draw, tag in zip(axes, [learning, confusion, roc], "abc"):
        draw(ax, small=True)
        ax.set_title(f"({tag})", fontsize=9, loc="left", color=INK)
    fig.tight_layout(w_pad=1.5); fig.savefig(FIG / "mlp_results_panel.png"); plt.close(fig)


if __name__ == "__main__":
    main()
