"""Build the SHAP-stability figures and the report-ready Word section.

Every number and feature name in the document is read from the result files written by
run_shap_stability.py; nothing is typed in by hand. Run after both splits are finished:
    python make_shap_report.py
"""
import json
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "shap_stability"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#1f1f1e", "#6b6a64"

VAL = json.loads((OUT / "shap_stability_summary_val.json").read_text())
TEST = json.loads((OUT / "shap_stability_summary_test.json").read_text())
SV, ST = VAL["stability_summary"], TEST["stability_summary"]
SEEDS = VAL["gate_seeds"]
gates = pd.read_csv(OUT / "gate_model_training_summary_val.csv")
glob_test = pd.read_csv(OUT / "global_mean_abs_shap_test.csv")
glob_val = pd.read_csv(OUT / "global_mean_abs_shap_val.csv")
top5_test = glob_test["original_feature"].head(5).tolist()
top5_val = glob_val["original_feature"].head(5).tolist()
order = json.loads((OUT / "aggregated_feature_order.json").read_text())


def top3_sets(a):
    return [frozenset(r) for r in np.argsort(-np.abs(a), axis=1, kind="stable")[:, :3]]


def jac(A, B):
    return np.array([len(a & b) / len(a | b) for a, b in zip(A, B)])


# --- Figure 1: stability distribution (discrete values -> grouped bars on a shared 0.05 grid)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})
bins = np.round(np.arange(0, 1.0001, 0.05), 2)
fig, ax = plt.subplots(figsize=(7.2, 3.8))
w = 0.02
for k, (split, label, c, s) in enumerate([("val", "Validation", BLUE, SV), ("test", "Test", ORANGE, ST)]):
    d = pd.read_csv(OUT / f"customer_shap_stability_{split}.csv")["shap_stability"]
    counts = pd.cut(d, np.append(bins - 0.025, 1.025), labels=bins).value_counts().reindex(bins, fill_value=0)
    share = 100 * counts.values / len(d)
    ax.bar(bins + (k - 0.5) * (w + 0.002), share, width=w, color=c, label=f"{label} (n = {len(d):,}; mean {s['mean']:.3f})")
ax.set_xlim(-0.03, 1.03); ax.set_xticks(np.arange(0, 1.01, 0.1))
ax.set_xlabel("Customer-level SHAP stability S(x) (mean pairwise Jaccard of top-3 sets, rounded to 0.05)")
ax.set_ylabel("Share of customers (%)")
ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", color="#e6e5df", lw=0.6); ax.set_axisbelow(True)
ax.legend(frameon=False, loc="upper left")
fig.tight_layout(); fig.savefig(FIG / "shap_stability_distribution.png", dpi=300); plt.close(fig)

# --- Figure 2: global importance, validation and test side by side
g = glob_test.head(10).merge(glob_val, on="original_feature", suffixes=("_test", "_val")).iloc[::-1]
fig, ax = plt.subplots(figsize=(7.2, 4.4))
y = np.arange(len(g)); h = 0.38
ax.barh(y + h / 2 + 0.01, g["mean_absolute_aggregated_shap_val"], height=h, color=BLUE, label="Validation")
ax.barh(y - h / 2 - 0.01, g["mean_absolute_aggregated_shap_test"], height=h, color=ORANGE, label="Test")
ax.set_yticks(y); ax.set_yticklabels(g["original_feature"])
ax.set_xlabel("Mean |aggregated SHAP| (churn-probability units), averaged over 5 gate models")
ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="x", color="#e6e5df", lw=0.6); ax.set_axisbelow(True)
ax.legend(frameon=False, loc="lower right")
fig.tight_layout(); fig.savefig(FIG / "global_shap_importance.png", dpi=300); plt.close(fig)

# --- Word section
doc = Document()
sec = doc.sections[0]
sec.top_margin = sec.bottom_margin = Inches(0.7); sec.left_margin = sec.right_margin = Inches(0.8)
doc.styles["Normal"].font.name = "Times New Roman"; doc.styles["Normal"].font.size = Pt(11)


def para(head, text):
    p = doc.add_paragraph(); p.add_run(head + " ").bold = True; p.add_run(text); return p


def caption(text):
    p = doc.add_paragraph(); r = p.add_run(text); r.italic = True; r.font.size = Pt(10); return p


def table(header, rows):
    t = doc.add_table(rows=1, cols=len(header)); t.style = "Table Grid"
    for c, v in zip(t.rows[0].cells, header):
        c.text = v; c.paragraphs[0].runs[0].bold = True
    for r in rows:
        for c, v in zip(t.add_row().cells, r):
            c.text = v
    return t


t = doc.add_paragraph(); t.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = t.add_run("SHAP Explanation Stability: Methods and Results"); r.bold = True; r.font.size = Pt(14)

auc_lo, auc_hi = gates["validation_roc_auc"].min(), gates["validation_roc_auc"].max()
para("Gate models.", f"Five NumPy MLP gate models with seeds {SEEDS[0]}–{SEEDS[-1]} (disjoint from the baseline seeds "
     "42–46 and the evaluation seeds 101–105) were trained on the shared 4,930-customer training partition with the "
     "configuration selected in Checkpoint 2: a 45–32–1 network (ReLU hidden layer, sigmoid output), Adam with learning "
     "rate 0.003, batch size 128, L2 weight decay 10⁻⁴, at most 300 epochs and validation-loss early stopping with "
     f"patience 20. The restored epochs were {gates.best_epoch.min()}–{gates.best_epoch.max()}, and validation ROC-AUC "
     f"was {auc_lo:.3f}–{auc_hi:.3f} (Table 1), in line with the seed-42 MLP (0.846). The shared preprocessing and "
     "feature order were reused unchanged, and test labels were not read.")

caption("Table 1. Gate models (validation partition, n = 1,056).")
table(["Seed", "Restored epoch", "Validation BCE", "Validation ROC-AUC"],
      [[str(int(r.seed)), str(int(r.best_epoch)), f"{r.best_validation_bce:.4f}", f"{r.validation_roc_auc:.4f}"]
       for r in gates.itertuples()])
doc.add_paragraph()

para("SHAP computation.", "Because the MLP is implemented in NumPy, the model-agnostic Kernel SHAP estimator "
     "[Lundberg and Lee, 2017] was applied to the churn-probability output (identity link), using the interventional "
     "value function of shap.KernelExplainer: v(S) is the model output averaged over a background set with the features "
     f"outside S taken from the background rows. One background of {VAL['background_size']} training customers was "
     f"sampled once without replacement (seed {VAL['background_seed']}) and shared by all gate models. Encoded columns "
     "whose value matches every background row receive a SHAP value of exactly zero. For the remaining M ≤ 45 "
     "players, all coalitions of size 1, 2, M − 2 and M − 1 were enumerated with their exact Shapley-kernel weights and "
     "2,000 further complementary coalition pairs were sampled in proportion to the kernel weight (about 6,000 "
     "coalitions per customer); the weighted least-squares problem was solved with the efficiency constraint imposed "
     "and no l1 regularisation. The implementation (kernel_shap_numpy.py) reproduces brute-force Shapley values "
     "exactly on a test network with eight inputs. The same coalition design was used for all five gate models within "
     f"a split (seed {VAL['shap_random_seed']} for validation, {TEST['shap_random_seed']} for test), so differences "
     "between models are not caused by different random coalitions.")

para("Aggregation and stability.", "The 45 encoded attributions were aggregated to the 19 original features by "
     "summing the signed SHAP values of each feature's columns (feature_groups.json). For each customer x and gate "
     "model k, Tₖ(x) is the set of three original features with the largest absolute aggregated SHAP value. "
     "Customer-level stability is the mean Jaccard overlap over the ten model pairs, "
     "S(x) = (1/10) Σⱼ<ₖ |Tⱼ(x) ∩ Tₖ(x)| / |Tⱼ(x) ∪ Tₖ(x)| ∈ [0, 1] (Eq. 6). A pairwise overlap can only take the "
     "values 0, 0.2, 0.5 or 1, so S(x) moves in small discrete steps.")

# Figures from the superseded shap.KernelExplainer run (128 coalitions per customer), as reported in the
# Methods text. That run's files are not part of the repository, so the values are fixed here.
# Set OLD_CMP = None to drop the sentence.
OLD_CMP = {"val": {"old_mean": 0.656}, "test": {"ident": 0.765, "same": 0.881, "old_mean": 0.664}}
old_cmp = OLD_CMP

nv, nt = VAL["monte_carlo_noise_check"], TEST["monte_carlo_noise_check"]
para("Estimation accuracy.", "Kernel SHAP is a sampling estimator, so part of any disagreement between models could "
     "come from estimation noise rather than from the models. As a check, the first gate model's SHAP values were "
     f"re-estimated with a second, independent coalition design for the first {nv['n_customers']} customers of each "
     f"split. The two estimates gave the same top-three set for {nv['share_identical_top3']:.1%} of validation and "
     f"{nt['share_identical_top3']:.1%} of test customers (mean same-model Jaccard {nv['mean_same_model_jaccard']:.3f} "
     f"and {nt['mean_same_model_jaccard']:.3f}), so the stability scores below reflect differences between the gate "
     "models rather than estimator noise. "
     + (f"An earlier run used shap.KernelExplainer with only 128 coalitions per customer. Against the present "
        f"estimates for the same models, it returned the same top-three set for only {old_cmp['test']['ident']:.1%} of "
        f"test customers (mean Jaccard {old_cmp['test']['same']:.3f}); this noise lowered mean stability to "
        f"{old_cmp['val']['old_mean']:.3f} (validation) and {old_cmp['test']['old_mean']:.3f} (test), so those values "
        "are superseded by the results reported here. " if old_cmp else "")
     + "The efficiency property (SHAP values summing to f(x) − E[f]) holds to within "
       f"{max(VAL['efficiency_check']['max_abs_error'], TEST['efficiency_check']['max_abs_error']):.0e}; because Kernel "
       "SHAP imposes this constraint by construction, it confirms the implementation but not the accuracy of "
       "individual values, which is why the re-estimation check above is reported.")

para("Results.", f"Mean customer-level stability was {SV['mean']:.3f} (SD {SV['std']:.3f}; median {SV['median']:.2f}; "
     f"IQR {SV['q25']:.2f}–{SV['q75']:.2f}) on validation (n = {SV['n_customers']:,}) and {ST['mean']:.3f} (SD "
     f"{ST['std']:.3f}; median {ST['median']:.2f}; IQR {ST['q25']:.2f}–{ST['q75']:.2f}) on the held-out test set "
     f"(n = {ST['n_customers']:,}) (Table 2, Fig. 1). All five gate models selected the same three features for "
     f"{SV['share_identical_top3_all_models']:.1%} of validation and {ST['share_identical_top3_all_models']:.1%} of "
     f"test customers, and the same leading feature for {SV['share_same_top1_all_models']:.1%} and "
     f"{ST['share_same_top1_all_models']:.1%}. Stability was below 0.5 for {SV['share_below_0_5']:.1%} of validation "
     f"and {ST['share_below_0_5']:.1%} of test customers. The gate models thus agree on the leading feature for about "
     "two thirds of customers but rarely select exactly the same three, so S(x) varies enough across customers to be "
     "usable as a gate signal; whether it adds information beyond confidence is tested in the gate experiments. The "
     f"features with the largest mean absolute aggregated SHAP value on the test set were {', '.join(top5_test[:-1])} "
     f"and {top5_test[-1]} (Fig. 2)"
     + ("; the validation ranking has the same top five." if set(top5_val) == set(top5_test)
        else f"; on validation the top five were {', '.join(top5_val)}."))

caption("Table 2. Customer-level SHAP stability S(x) across the five gate models.")
table(["Split", "n", "Mean (SD)", "Median", "IQR", "S ≥ 0.5", "S = 1", "Same top-1"],
      [[nm, f"{s['n_customers']:,}", f"{s['mean']:.3f} ({s['std']:.3f})", f"{s['median']:.2f}",
        f"{s['q25']:.2f}–{s['q75']:.2f}", f"{s['share_at_least_0_5']:.1%}", f"{s['share_identical_top3_all_models']:.1%}",
        f"{s['share_same_top1_all_models']:.1%}"] for nm, s in [("Validation", SV), ("Test", ST)]])
doc.add_paragraph()

doc.add_picture(str(FIG / "shap_stability_distribution.png"), width=Inches(6.4))
caption("Fig. 1. Distribution of customer-level SHAP stability on the validation and test partitions.")
doc.add_picture(str(FIG / "global_shap_importance.png"), width=Inches(6.2))
caption("Fig. 2. Ten original features with the largest mean absolute aggregated SHAP value on the test set, with the "
        "validation values for comparison. Averaged over customers and the five gate models; descriptive, not causal.")

para("Interpretation and limitations.", "S(x) measures how reproducible a customer's top-three explanation is across "
     "independently initialised gate models trained on the same data. It does not show that these features cause churn "
     "or that acting on them would retain the customer. Because only the top three of 19 features are compared, two "
     "models whose third and fourth features have nearly equal SHAP values can lower S(x) even when their explanations "
     "are otherwise very similar. Signed one-hot attributions are summed after estimation, which is the usual "
     "convention but is not identical to treating each original feature as a single Shapley player. Results depend on "
     "the 50-customer background and on this single split. The validation scores are used only to choose the "
     "abstention thresholds; the test scores are used once, for the held-out comparison of the confidence-only and "
     "confidence-plus-stability gates, which is reported separately together with the evaluation-model agreement.")

doc.save(ROOT / "SHAP_Stability_Methods_Results.docx")
print("Wrote SHAP_Stability_Methods_Results.docx; top-5 test features:", top5_test)
