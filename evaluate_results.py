"""CAI 421 - Student 4: final evaluation, comparison with published studies, tables and figures.

Reads (nothing is retrained except the two reference baselines, which are refitted exactly as in
Checkpoint 2 so that paired comparisons are possible):
  --processed  data/processed/              shared splits (y_test used for evaluation only)
  --s2         SHAP-stability package       results/shap_stability/ (gate SHAP + probabilities)
  --s3         abstention-gate package      data/gate_inputs/, results/gate/ (frozen thresholds, results)
Writes tables/*.csv, figures/*.png and results/results_summary.json.

    python evaluate_results.py --processed <dir> --s2 <SHAP pkg> --s3 <gate pkg>
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)

from mlp import train

ROOT = Path(__file__).resolve().parent
TAB, FIG, RES = ROOT / "tables", ROOT / "figures", ROOT / "results"
SEED, N_BOOT = 42, 2000
GATE_SEEDS = [1, 2, 3, 4, 5]
BLUE, ORANGE, AQUA, GRAY, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#8a897f", "#1f1f1e", "#6b6a64", "#e6e5df"

# Action map exactly as in Student 3's build_gate_inputs.py (used only to rebuild per-gate-model actions
# for the diagnostic in Table 4; the gate itself is evaluated from Student 3's saved inputs).
ACTIONS = ["CONTRACT_OFFER", "PLAN_REVIEW", "ADDON_SUPPORT_BUNDLE", "STREAMING_OFFER", "BILLING_REVIEW"]


def actions_from_shap(phi, order, f2a):
    cols = [j for j, f in enumerate(order) if f in f2a]
    sub = phi[:, cols]
    best = sub.argmax(axis=1)
    ids = np.array([ACTIONS.index(f2a[order[cols[b]]]) for b in best])
    return np.where(sub[np.arange(len(sub)), best] > 0, ids, -1)


def pairwise_agreement(E):
    return np.mean([E[:, i] == E[:, j] for i, j in combinations(range(E.shape[1]), 2)], axis=0)


def clf_metrics(y, p):
    yhat = p >= 0.5
    tn, fp, fn, tp = confusion_matrix(y, yhat).ravel()
    return {"accuracy": accuracy_score(y, yhat), "precision": precision_score(y, yhat), "recall": recall_score(y, yhat),
            "f1": f1_score(y, yhat), "roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
            "brier": brier_score_loss(y, p), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def paired_boot_diff(y, pa, pb, keys=("roc_auc", "pr_auc", "f1", "brier")):
    """Class-stratified paired bootstrap of metric(pa) - metric(pb), as in Checkpoint 2."""
    rng = np.random.default_rng(SEED)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    out = {k: [] for k in keys}
    for _ in range(N_BOOT):
        i = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        ma, mb = clf_metrics(y[i], pa[i]), clf_metrics(y[i], pb[i])
        for k in keys:
            out[k].append(ma[k] - mb[k])
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in out.items()}


def verify_gate_inputs(s2_dir: Path, gate_inputs: Path, processed: Path) -> dict:
    """Check that Student 3's gate inputs are the customer-level outputs of Student 2's final run.

    For each split: row counts, row order (raw_row_index vs the shared split index), then the stability
    S(x) and the five gate-model probabilities are compared with np.testing.assert_allclose.
    Stops with a clear message if anything differs; nothing is modified.
    """
    report = {}
    s2_res = s2_dir / "results" / "shap_stability"
    for split in ("val", "test"):
        def fail(msg):
            raise SystemExit(f"INPUT CHECK FAILED [{split}]: {msg}\n  Student 2 file: "
                             f"{s2_res / f'customer_shap_stability_{split}.csv'}\n  Student 3 folder: {gate_inputs}")
        s2 = pd.read_csv(s2_res / f"customer_shap_stability_{split}.csv")
        idx = pd.read_csv(processed / f"idx_{split}.csv").iloc[:, 0].to_numpy()
        y = pd.read_csv(processed / f"y_{split}.csv").iloc[:, 0].to_numpy()
        s3_S = np.load(gate_inputs / f"stability_{split}.npy").astype(float)
        s3_P = np.load(gate_inputs / f"gate_probs_{split}.npy").astype(float)
        n = len(y)
        # 1) row counts
        counts = {"labels": n, "split_index": len(idx), "student2_rows": len(s2), "student3_stability": len(s3_S),
                  "student3_probabilities": len(s3_P)}
        if len(set(counts.values())) != 1:
            fail(f"row counts differ: {counts}")
        # 2) ordering and customer alignment with the shared split
        if not np.array_equal(s2["row_position"].to_numpy(), np.arange(n)):
            fail("Student 2 row_position is not 0..n-1 in order")
        if not np.array_equal(s2["raw_row_index"].to_numpy(), idx):
            bad = int(np.argmax(s2["raw_row_index"].to_numpy() != idx))
            fail(f"customer order differs from idx_{split}.csv (first mismatch at row {bad})")
        if s3_P.shape != (n, len(GATE_SEEDS)):
            fail(f"gate_probs_{split}.npy has shape {s3_P.shape}, expected {(n, len(GATE_SEEDS))}")
        # 3) values: stability, and the per-seed probabilities as an independent alignment check
        try:
            np.testing.assert_allclose(s3_S, s2["shap_stability"].to_numpy(float), rtol=0, atol=1e-9,
                                       err_msg=f"stability_{split}.npy (Student 3) != shap_stability (Student 2)")
            np.testing.assert_allclose(s3_P, s2[[f"seed_{k}_probability" for k in GATE_SEEDS]].to_numpy(float),
                                       rtol=0, atol=1e-9,
                                       err_msg=f"gate_probs_{split}.npy (Student 3) != seed probabilities (Student 2)")
        except AssertionError as e:
            fail(str(e).strip())
        report[split] = {"n": n, "order_matches_split_index": True, "stability_max_abs_diff": float(np.max(np.abs(
            s3_S - s2["shap_stability"].to_numpy(float)))), "stability_mean": float(s3_S.mean())}
        print(f"[check] {split}: {n} customers, order matches idx_{split}.csv, stability identical "
              f"(mean {s3_S.mean():.5f}), gate probabilities identical", flush=True)
    return report


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", type=Path, required=True)
    ap.add_argument("--s2", type=Path, required=True)
    ap.add_argument("--s3", type=Path, required=True)
    a = ap.parse_args()
    for d in (TAB, FIG, RES):
        d.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})

    # ------------------------------------------------------------------ inputs
    L = lambda f: pd.read_csv(a.processed / f)
    Xtr, Xva, Xte = (L(f"X_{s}.csv").to_numpy(np.float64) for s in ("train", "val", "test"))
    ytr, yva, yte = (L(f"y_{s}.csv").iloc[:, 0].to_numpy(int) for s in ("train", "val", "test"))
    gi = a.s3 / "data" / "gate_inputs"
    P = np.load(gi / "gate_probs_test.npy"); S = np.load(gi / "stability_test.npy").astype(float)
    act = np.load(gi / "gate_action_test.npy").astype(int); E = np.load(gi / "eval_actions_test.npy").astype(int)
    amap = json.loads((gi / "action_map.json").read_text())
    gate_res = json.loads((a.s3 / "results" / "gate" / "gate_results.json").read_text())
    thr = json.loads((a.s3 / "results" / "gate" / "thresholds.json").read_text())
    input_check = verify_gate_inputs(a.s2, gi, a.processed)
    (RES / "input_verification.json").write_text(json.dumps(input_check, indent=2))
    pbar = P.mean(axis=1)
    agr = pairwise_agreement(E)
    unan = (E == E[:, [0]]).all(axis=1)
    n = len(yte)
    assert P.shape == (n, 5) and E.shape == (n, 5) and len(S) == n

    # consistency check: Student 3's stored results are reproduced from the saved inputs
    def rec(tp, ts):
        return (act >= 0) & (pbar >= tp - 1e-9) & (S >= ts - 1e-9)
    for c, t in gate_res["targets"].items():
        for sysname, mask in (("B3", rec(t["thresholds"]["b3_tau_p"], 0)),
                              ("gate", rec(t["thresholds"]["gate_tau_p"], t["thresholds"]["gate_tau_s"]))):
            assert abs(agr[mask].mean() - t[sysname]["agreement"]) < 1e-9, (c, sysname)
            assert mask.sum() == t[sysname]["n_recommended"], (c, sysname)

    # ------------------------------------------------------------------ Table 1: predictive performance
    lr = LogisticRegression(C=1.0, max_iter=10_000, tol=1e-4, random_state=42).fit(Xtr, ytr)
    p_b1 = lr.predict_proba(Xte)[:, 1]
    mlp, _, best = train(Xtr, ytr, Xva, yva, hidden=(32,), seed=42, lr=0.003, batch_size=128, weight_decay=1e-4,
                         max_epochs=300, patience=20)
    p_mlp = mlp.predict_proba(Xte)
    singles = [clf_metrics(yte, P[:, k]) for k in range(5)]
    rows = {"B1 logistic regression": clf_metrics(yte, p_b1), "MLP, seed 42 (Checkpoint 2)": clf_metrics(yte, p_mlp),
            "Gate ensemble (mean of 5 MLPs)": clf_metrics(yte, pbar)}
    t1 = pd.DataFrame(rows).T
    sm = pd.DataFrame(singles)
    t1.loc["Single gate MLPs, mean ± SD"] = [f"{sm[c].mean():.3f} ± {sm[c].std(ddof=1):.3f}" if c not in ("tn", "fp", "fn", "tp")
                                             else "" for c in t1.columns]
    t1.to_csv(TAB / "table1_predictive_performance.csv")
    ci_ens_b1 = paired_boot_diff(yte, pbar, p_b1)
    ci_ens_mlp = paired_boot_diff(yte, pbar, p_mlp)

    # ------------------------------------------------------------------ Table 2: policies (Student 3, frozen)
    pol = [{"target": "–", "system": "B2 (no abstention)", "tau_p": "–", "tau_s": "–", **gate_res["B2"]}]
    for c, t in gate_res["targets"].items():
        th = t["thresholds"]
        pol.append({"target": f"{float(c):.0%}", "system": "B3 confidence-only", "tau_p": th["b3_tau_p"], "tau_s": "–", **t["B3"]})
        pol.append({"target": f"{float(c):.0%}", "system": "Proposed confidence + stability", "tau_p": th["gate_tau_p"],
                    "tau_s": th["gate_tau_s"], **t["gate"]})
    t2 = pd.DataFrame(pol)
    t2.to_csv(TAB / "table2_policy_comparison.csv", index=False)
    t3 = pd.DataFrame([{"target": f"{float(c):.0%}", "metric": k, "difference_gate_minus_B3": t["diff_gate_minus_B3"][k],
                        "ci95_low": t["ci95"][k][0], "ci95_high": t["ci95"][k][1],
                        "ci_excludes_zero": not (t["ci95"][k][0] <= 0 <= t["ci95"][k][1])}
                       for c, t in gate_res["targets"].items() for k in t["diff_gate_minus_B3"]])
    t3.to_csv(TAB / "table3_gate_minus_B3_bootstrap.csv", index=False)

    # ------------------------------------------------------------------ Table 4: why — diagnostics
    order = amap["feature_order"]; f2a = amap["feature_to_action"]
    shap_dir = a.s2 / "results" / "shap_stability"
    phis = [np.load(shap_dir / f"aggregated_shap_test_seed_{s}.npy") for s in GATE_SEEDS]
    G = np.stack([actions_from_shap(p, order, f2a) for p in phis], axis=1)      # per-gate-model actions
    gate_action_agr = pairwise_agreement(G)
    m = act >= 0
    rho = lambda u, v: float(spearmanr(u[m], v[m]).statistic)
    diag = {"n_actionable": int(m.sum()),
            "spearman_S_vs_eval_agreement": rho(S, agr),
            "spearman_pbar_vs_eval_agreement": rho(pbar, agr),
            "spearman_S_vs_pbar": rho(S, pbar),
            "spearman_gate_action_agreement_vs_eval_agreement": rho(gate_action_agr, agr),
            "gate_action_equals_eval_action_share": float(np.mean(E[m] == act[m][:, None]))}
    pd.Series(diag).to_csv(TAB / "table4_diagnostics.csv", header=["value"])

    bands = [(0, .5, "< 0.50"), (.5, .6, "0.50–\n0.59"), (.6, .7, "0.60–\n0.69"), (.7, .8, "0.70–\n0.79"), (.8, 1.01, "≥ 0.80")]
    by_s = pd.DataFrame([{"stability_band": lab, "n": int(((S >= lo) & (S < hi) & m).sum()),
                          "eval_agreement": agr[(S >= lo) & (S < hi) & m].mean(),
                          "mean_pbar": pbar[(S >= lo) & (S < hi) & m].mean(),
                          "nonchurn_share": (yte[(S >= lo) & (S < hi) & m] == 0).mean()} for lo, hi, lab in bands])
    q = pd.qcut(pbar[m], 5, labels=False)
    by_p = pd.DataFrame([{"pbar_quintile": f"Q{k + 1}", "pbar_range": f"{pbar[m][q == k].min():.3f}–{pbar[m][q == k].max():.3f}",
                          "n": int((q == k).sum()), "eval_agreement": agr[m][q == k].mean(),
                          "mean_stability": S[m][q == k].mean(), "nonchurn_share": (yte[m][q == k] == 0).mean()}
                         for k in range(5)])
    by_s.assign(stability_band=by_s.stability_band.str.replace("\n", "")).to_csv(TAB / "table4a_agreement_by_stability.csv", index=False)
    by_p.to_csv(TAB / "table4b_agreement_by_confidence.csv", index=False)

    # ------------------------------------------------------------------ Table 5: published studies
    imp = np.stack([np.abs(p).mean(axis=0) for p in phis])                     # global importance per gate model
    rank_rho = [float(spearmanr(imp[i], imp[j]).statistic) for i, j in combinations(range(5), 2)]
    pred = P >= 0.5
    pred_disagree = float(np.mean(pred.any(axis=1) & ~pred.all(axis=1)))
    b2 = gate_res["B2"]
    lit = {"global_rank_spearman_min": min(rank_rho), "global_rank_spearman_max": max(rank_rho),
           "customer_top3_stability_mean": float(S.mean()),
           "gate_models_class_disagreement_share": pred_disagree,
           "eval_models_action_not_unanimous_share_B2": 1 - b2["unanimous"],
           "b1_roc_auc": rows["B1 logistic regression"]["roc_auc"]}
    t5 = pd.DataFrame([
        {"study": "[1] Vutla et al. (2026)", "data": "IBM Telco (+ Iranian, Cell2Cell)",
         "their_measure": "Global SHAP feature-ranking agreement across 5 CV folds (Spearman/Kendall); single training seed",
         "their_result": "LR ROC-AUC 0.8397; ranking agreement 0.86–1.00; ROI 48.0% under assumed $30 cost and 30% success",
         "our_measure": "Same dataset; global ranking agreement across 5 seeds; per-customer top-3 Jaccard",
         "our_result": f"LR ROC-AUC {lit['b1_roc_auc']:.3f}; global ranking Spearman {min(rank_rho):.2f}–{max(rank_rho):.2f}; "
                       f"customer-level stability mean {S.mean():.3f}"},
        {"study": "[2] Black et al. (2022)", "data": "7 non-churn benchmarks (credit, income, medical, images)",
         "their_measure": "Prediction disagreement across seeds; selective ensembles abstaining by hypothesis test",
         "their_result": "Single models disagree on up to 57% of points; 10+ model selective ensembles: 0 inconsistent "
                         "predictions, abstention as low as 1.5%",
         "our_measure": "Class disagreement of 5 gate MLPs at 0.5; action disagreement of 5 evaluation MLPs",
         "our_result": f"Class disagreement {pred_disagree:.1%} of test customers; actions not unanimous for "
                       f"{1 - b2['unanimous']:.1%} of B2 recommendations"}])
    t5.to_csv(TAB / "table5_published_comparison.csv", index=False)

    # ------------------------------------------------------------------ Figures
    # Fig A: agreement and outreach-to-nonchurn vs coverage (descriptive sweep on test; selection was on validation)
    order_conf = np.argsort(-pbar[m], kind="stable")
    order_stab = np.lexsort((-pbar[m], -S[m]))                                  # stability first, ties by confidence
    ks = np.arange(50, m.sum() + 1, 5)
    cov = ks / n

    def curve(o, vec):
        cs = np.cumsum(vec[m][o]); return cs[ks - 1] / ks

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))
    for ax, vec, ylab, key in ((axes[0], agr, "Evaluation-model action agreement", "agreement"),
                               (axes[1], (yte == 0).astype(float), "Outreach to non-churners", "outreach_to_nonchurn")):
        ax.plot(cov, curve(order_conf, vec), color=BLUE, lw=2, label="Ranked by mean churn probability p̄(x)")
        ax.plot(cov, curve(order_stab, vec), color=ORANGE, lw=2, label="Ranked by stability S(x)")
        ax.axhline(vec[m].mean(), color=GRAY, lw=1, ls="--", label="B2 (all actionable)")
        for c, t in gate_res["targets"].items():
            ax.scatter(t["B3"]["coverage"], t["B3"][key], s=42, color=BLUE, edgecolor="white", lw=1.5, zorder=3, marker="o")
            ax.scatter(t["gate"]["coverage"], t["gate"][key], s=48, color=ORANGE, edgecolor="white", lw=1.5, zorder=3, marker="D")
        ax.set_xlabel("Coverage (share of test customers)"); ax.set_ylabel(ylab); style(ax)
        ax.set_xlim(0, 0.95)
    from matplotlib.lines import Line2D
    h, l = axes[0].get_legend_handles_labels()
    h += [Line2D([], [], ls="", marker="o", color=BLUE, ms=6), Line2D([], [], ls="", marker="D", color=ORANGE, ms=6)]
    l += ["B3, frozen thresholds", "Proposed gate, frozen thresholds"]
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.995), fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.86)); fig.savefig(FIG / "fig_coverage_curves.png", dpi=300); plt.close(fig)

    # Fig B: agreement by stability band and by confidence quintile
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for ax, df, xcol, c, xl in ((axes[0], by_s, "stability_band", ORANGE, "Explanation stability S(x)"),
                                (axes[1], by_p, "pbar_quintile", BLUE, "Mean churn probability p̄(x) quintile (low → high)")):
        bars = ax.bar(df[xcol], df["eval_agreement"], color=c, width=0.6)
        for b, k in zip(bars, df["n"]):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f"{b.get_height():.2f}\nn={k}", ha="center",
                    va="bottom", fontsize=7, color=INK)
        ax.set_xlabel(xl); style(ax); ax.set_ylim(0, 1.12)
    axes[0].set_ylabel("Evaluation-model action agreement")
    fig.tight_layout(); fig.savefig(FIG / "fig_agreement_by_signal.png", dpi=300); plt.close(fig)

    # Fig C: gate − B3 differences with 95% CI
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for ax, k, title in ((axes[0], "agreement", "Action agreement"), (axes[1], "outreach_to_nonchurn", "Outreach to non-churners")):
        sub = t3[t3.metric == k].reset_index(drop=True)
        x = np.arange(len(sub))
        ax.errorbar(x, sub.difference_gate_minus_B3, yerr=[sub.difference_gate_minus_B3 - sub.ci95_low,
                    sub.ci95_high - sub.difference_gate_minus_B3], fmt="o", color=INK, ms=6, capsize=4, lw=1.5)
        ax.axhline(0, color=GRAY, lw=1)
        ax.set_xticks(x); ax.set_xticklabels([f"{t} target" for t in sub.target]); ax.set_xlim(-0.5, len(sub) - 0.5)
        ax.set_title(f"{title}: gate − B3", fontsize=9, color=INK); style(ax)
    axes[0].set_ylabel("Difference (95% bootstrap CI)")
    fig.tight_layout(); fig.savefig(FIG / "fig_gate_minus_b3.png", dpi=300); plt.close(fig)

    # ------------------------------------------------------------------ summary
    summary = {"n_test": n, "baseline_mlp_best_epoch": best, "table1": {k: {kk: float(vv) for kk, vv in v.items()}
               for k, v in rows.items()}, "single_gate_mlps_mean": sm.mean().to_dict(), "single_gate_mlps_sd": sm.std(ddof=1).to_dict(),
               "ci95_ensemble_minus_B1": ci_ens_b1, "ci95_ensemble_minus_mlp42": ci_ens_mlp,
               "thresholds": thr, "gate_results": gate_res, "diagnostics": diag, "literature": lit,
               "agreement_by_stability": by_s.to_dict("records"), "agreement_by_confidence": by_p.to_dict("records")}
    (RES / "results_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps({"diagnostics": diag, "literature": lit, "ci_ens_b1": ci_ens_b1}, indent=2))


if __name__ == "__main__":
    main()
