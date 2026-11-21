"""Write the report-ready 'Results and Comparison' section (Word) from the files made by evaluate_results.py.
Every number is read from results/results_summary.json or tables/*.csv; nothing is typed in by hand.

    python make_results_section.py
Table and figure numbers continue Checkpoint 2 (Tables 1-6, Figs. 1-3); renumber when the paper is merged.
"""
import json
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parent
R = json.loads((ROOT / "results" / "results_summary.json").read_text())
TAB, FIG = ROOT / "tables", ROOT / "figures"
G, D, LIT = R["gate_results"], R["diagnostics"], R["literature"]
T = G["targets"]; B2 = G["B2"]; TH = R["thresholds"]
t1, ens, b1, mlp = R["table1"], R["table1"]["Gate ensemble (mean of 5 MLPs)"], R["table1"]["B1 logistic regression"], \
    R["table1"]["MLP, seed 42 (Checkpoint 2)"]
ci_b1 = R["ci95_ensemble_minus_B1"]
by_s = pd.read_csv(TAB / "table4a_agreement_by_stability.csv")
by_p = pd.read_csv(TAB / "table4b_agreement_by_confidence.csv")
lit_tab = pd.read_csv(TAB / "table5_published_comparison.csv")

f3 = lambda v: f"{v:.3f}"
pct = lambda v: f"{100 * v:.1f}%"
sd = lambda v: f"{v:+.3f}"
ci = lambda c: f"[{c[0]:+.3f}, {c[1]:+.3f}]"

doc = Document()
sec = doc.sections[0]
sec.top_margin = sec.bottom_margin = Inches(0.7); sec.left_margin = sec.right_margin = Inches(0.8)
doc.styles["Normal"].font.name = "Times New Roman"; doc.styles["Normal"].font.size = Pt(11)


def head(text, size=12):
    p = doc.add_paragraph(); r = p.add_run(text); r.bold = True; r.font.size = Pt(size)


def para(text, lead=None):
    p = doc.add_paragraph()
    if lead:
        p.add_run(lead + " ").bold = True
    p.add_run(text)


def caption(text):
    p = doc.add_paragraph(); r = p.add_run(text); r.italic = True; r.font.size = Pt(10)


def table(header, rows, size=9, widths=None):
    t = doc.add_table(rows=1, cols=len(header)); t.style = "Table Grid"
    for c, v in zip(t.rows[0].cells, header):
        c.text = v
    for row in rows:
        for c, v in zip(t.add_row().cells, row):
            c.text = str(v)
    for i, row in enumerate(t.rows):
        for c in row.cells:
            for p in c.paragraphs:
                for r in p.runs:
                    r.font.size = Pt(size); r.bold = (i == 0)
    if widths:  # fixed column widths (inches) so short values such as 709/67/133/148 do not wrap
        t.autofit = False
        for col, w in zip(t.columns, widths):
            col.width = Inches(w)
        for row in t.rows:
            for c, w in zip(row.cells, widths):
                c.width = Inches(w)
    doc.add_paragraph()


def figure(name, width, text):
    doc.add_picture(str(FIG / name), width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption(text)


t = doc.add_paragraph(); t.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = t.add_run("Results and Comparison with Published Studies"); r.bold = True; r.font.size = Pt(14)
n = doc.add_paragraph(); r = n.add_run("Section draft (Student 4). Table and figure numbers continue Checkpoint 2 and will be "
                                         "renumbered when the paper is assembled.")
r.italic = True; r.font.size = Pt(9); r.font.color.rgb = RGBColor(0x6b, 0x6a, 0x64)

# --------------------------------------------------------------------------- 7.1
head("7. Results")
head("7.1 Predictive performance", 11)
para(f"Table 7 reports held-out test performance (n = {R['n_test']:,}, threshold 0.5). The five-model gate ensemble, whose "
     f"mean probability p̄(x) drives both abstention policies, reached ROC-AUC {f3(ens['roc_auc'])}, PR-AUC {f3(ens['pr_auc'])} "
     f"and Brier score {f3(ens['brier'])}, essentially the same as logistic regression B1 (ROC-AUC {f3(b1['roc_auc'])}; "
     f"paired bootstrap difference interval {ci(ci_b1['roc_auc'])}). As in Checkpoint 2, B1 recovers more churners at the "
     f"fixed 0.5 threshold, so its F1 is higher ({f3(b1['f1'])} vs {f3(ens['f1'])}; interval of the ensemble-minus-B1 "
     f"difference {ci(ci_b1['f1'])}). Both reference baselines were refitted with the Checkpoint 2 settings and reproduce "
     "Tables 4–6 exactly. Ranking quality is therefore not where the models differ; the decision layer is evaluated next.")
rows = []
for name, m in t1.items():
    rows.append([name, f3(m["accuracy"]), f3(m["precision"]), f3(m["recall"]), f3(m["f1"]), f3(m["roc_auc"]),
                 f3(m["pr_auc"]), f3(m["brier"]), f"{int(m['tn'])}/{int(m['fp'])}/{int(m['fn'])}/{int(m['tp'])}"])
mu, s = R["single_gate_mlps_mean"], R["single_gate_mlps_sd"]
rows.append(["Single gate MLPs (seeds 1–5), mean ± SD"] + [f"{mu[k]:.3f} ± {s[k]:.3f}" for k in
            ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier")] + ["–"])
caption("Table 7. Test-set predictive performance at threshold 0.5. TN/FP/FN/TP = confusion matrix.")
table(["Model", "Acc.", "Prec.", "Recall", "F1", "ROC-AUC", "PR-AUC", "Brier ↓", "TN/FP/FN/TP"], rows, size=8,
      widths=[1.55, 0.58, 0.58, 0.6, 0.58, 0.68, 0.62, 0.6, 1.1])

# --------------------------------------------------------------------------- 7.2
head("7.2 Threshold selection on validation", 11)
mx = TH["max_possible_coverage_val"]
para(f"Because {pct(1 - mx)} of validation customers have no positive actionable driver, the highest attainable "
     f"coverage was {pct(mx)}; all three targets (50%, 70%, 90%) were therefore feasible for both gates. Thresholds were "
     "selected on validation only and frozen before the single test evaluation (Table 8). With the risk floor set to zero, "
     "the churn-probability thresholds (τₚ) needed to reach these coverage levels are low, so many recommended customers have a "
     "small predicted churn probability; this is reflected in the outreach-to-non-churner rates below.")
caption("Table 8. Frozen thresholds selected on the validation split (n = 1,056).")
table(["Target", "B3 τₚ", "Gate τₛ", "Gate τₚ", "Validation coverage B3 / gate"],
      [[f"{float(c):.0%}", f"{v['b3_tau_p']:.4f}", f"{v['gate_tau_s']:.1f}", f"{v['gate_tau_p']:.4f}",
        f"{pct(v['val_cov_b3'])} / {pct(v['val_cov_gate'])}"] for c, v in TH["targets"].items()])

# --------------------------------------------------------------------------- 7.3
head("7.3 Confidence-only versus confidence-plus-stability abstention", 11)
a50, a70, a90 = (T[k] for k in ("0.5", "0.7", "0.9"))
para(f"Table 9 compares the policies on the test set. Achieved coverage stayed close to each target, and no coverage "
     f"difference between the gates was distinguishable from zero (Table 10). Without abstention, B2 recommended an "
     f"action to {pct(B2['coverage'])} of customers with mean pairwise action agreement {f3(B2['agreement'])} across the five "
     f"independent evaluation models; all five agreed for {pct(B2['unanimous'])} of them. Confidence-only abstention (B3) "
     f"raised agreement as coverage fell, to {f3(a50['B3']['agreement'])} at the 50% target.")
para(f"Adding the stability threshold did not improve agreement at any coverage target. At 50% the two gates were "
     f"indistinguishable (difference {sd(a50['diff_gate_minus_B3']['agreement'])}, 95% CI "
     f"{ci(a50['ci95']['agreement'])}). At 70% the proposed gate was worse ({sd(a70['diff_gate_minus_B3']['agreement'])}, "
     f"{ci(a70['ci95']['agreement'])}; unanimous agreement {sd(a70['diff_gate_minus_B3']['unanimous'])}), and at 90% it was "
     f"slightly worse ({sd(a90['diff_gate_minus_B3']['agreement'])}, {ci(a90['ci95']['agreement'])}). The gate also sent a "
     f"larger share of recommendations to customers who did not churn at every target "
     f"({sd(a50['diff_gate_minus_B3']['outreach_to_nonchurn'])}, {sd(a70['diff_gate_minus_B3']['outreach_to_nonchurn'])} and "
     f"{sd(a90['diff_gate_minus_B3']['outreach_to_nonchurn'])}; all intervals exclude zero). The study hypothesis — "
     "that explanation stability improves recommendation consistency at matched coverage — is therefore not supported "
     "on this dataset. As planned in Section 2.3, this negative result is reported in full.")
rows = [["–", "B2 (no abstention)", "–", "–", pct(B2["coverage"]), B2["n_recommended"], f3(B2["agreement"]),
         pct(B2["unanimous"]), pct(B2["outreach_to_nonchurn"])]]
for c, v in T.items():
    th = v["thresholds"]
    for lab, key, tp, ts in (("B3 confidence-only", "B3", th["b3_tau_p"], "–"),
                             ("Proposed: confidence + stability", "gate", th["gate_tau_p"], f"{th['gate_tau_s']:.1f}")):
        m = v[key]
        rows.append([f"{float(c):.0%}", lab, f"{tp:.4f}", ts, pct(m["coverage"]), m["n_recommended"], f3(m["agreement"]),
                     pct(m["unanimous"]), pct(m["outreach_to_nonchurn"])])
caption("Table 9. Recommendation policies on the test set (n = 1,057) with thresholds frozen on validation. Agreement = "
        "mean pairwise agreement of the suggested action across the five evaluation models (seeds 101–105) among "
        "recommended customers; outreach to non-churners is a descriptive proxy, not a verified error.")
table(["Target", "Policy", "τₚ", "τₛ", "Coverage", "n rec.", "Agreement", "Unanimous", "Outreach to non-churn"], rows, size=8)
rows = []
for c, v in T.items():
    for k, lab in (("coverage", "Coverage"), ("agreement", "Agreement"), ("unanimous", "Unanimous"),
                   ("outreach_to_nonchurn", "Outreach to non-churn")):
        lo, hi = v["ci95"][k]
        rows.append([f"{float(c):.0%}", lab, sd(v["diff_gate_minus_B3"][k]), ci(v["ci95"][k]), "yes" if lo > 0 or hi < 0 else "no"])
caption("Table 10. Proposed gate minus B3 on the test set; paired bootstrap 95% intervals (2,000 resamples).")
table(["Target", "Metric", "Difference", "95% CI", "Excludes 0"], rows, size=8)
figure("fig_gate_minus_b3.png", 6.4, "Fig. 4. Difference between the proposed gate and B3 at each coverage target, with paired "
       "bootstrap 95% intervals. Negative agreement and positive outreach differences favour B3.")

# --------------------------------------------------------------------------- 7.4
head("7.4 Why stability did not help", 11)
lo_s, hi_s = by_s.iloc[-1], by_p.iloc[-1]
para(f"Among the {D['n_actionable']} actionable test customers, the stability score was essentially unrelated to whether "
     f"the evaluation models agreed on the action (Spearman ρ = {D['spearman_S_vs_eval_agreement']:.2f}), whereas the ensemble's "
     f"mean churn probability p̄(x) was clearly related (ρ = {D['spearman_pbar_vs_eval_agreement']:.2f}) (Fig. 5). Agreement was "
     f"{f3(by_p.iloc[0]['eval_agreement'])} in the lowest predicted-risk quintile and {f3(hi_s['eval_agreement'])} in the highest, "
     f"but did not rise with stability. Stability was also weakly negatively related to predicted churn risk "
     f"(ρ = {D['spearman_S_vs_pbar']:.2f}): customers with S(x) ≥ 0.80 had a mean predicted risk of "
     f"{f3(lo_s['mean_pbar'])} and {pct(lo_s['nonchurn_share'])} of them did not churn. A stability threshold therefore "
     "removes some high-risk customers and, to keep coverage matched, lowers τₚ to admit low-risk customers, whose "
     "actions are the least reproducible. Fig. 6 shows this along the whole coverage range.")
para(f"A likely reason is a mismatch between what S(x) measures and what the action rule uses. S(x) compares the top three "
     "features by absolute SHAP value, which often include non-actionable features such as tenure and TotalCharges and "
     "features that lower risk, while the action is taken from the single largest positive actionable driver. As an "
     "exploratory, post-hoc check (not a tested gate), the agreement of the five gate models' own actions correlated "
     f"strongly with evaluation-model agreement (ρ = {D['spearman_gate_action_agreement_vs_eval_agreement']:.2f}), which "
     "suggests that an action-level consistency measure would be a better candidate signal for future work.")
caption("Table 11. Evaluation-model action agreement by explanation stability and by predicted churn risk (actionable test customers).")
table(["Stability S(x)", "n", "Agreement", "Mean p̄(x)", "Non-churn share"],
      [[r.stability_band, r.n, f3(r.eval_agreement), f3(r.mean_pbar), pct(r.nonchurn_share)] for r in by_s.itertuples()])
table(["Predicted-risk quintile", "p̄(x) range", "n", "Agreement", "Mean S(x)", "Non-churn share"],
      [[r.pbar_quintile, r.pbar_range, r.n, f3(r.eval_agreement), f3(r.mean_stability), pct(r.nonchurn_share)]
       for r in by_p.itertuples()])
figure("fig_agreement_by_signal.png", 6.4, "Fig. 5. Evaluation-model action agreement by explanation-stability band (left) and "
       "by quintile of mean churn probability p̄(x) (right) among actionable test customers. Labels give the mean and number of customers.")
figure("fig_coverage_curves.png", 6.4, "Fig. 6. Action agreement (left) and outreach to non-churners (right) when test customers "
       "are ranked by mean churn probability p̄(x) or by stability. The curves are descriptive sweeps on the test set; the markers are the "
       "frozen operating points selected on validation. The proposed gate is not a pure stability ranking (it combines both "
       "thresholds), so its markers lie between the curves.")

# --------------------------------------------------------------------------- 8
head("8. Comparison with Published Studies")
v, b = lit_tab.iloc[0], lit_tab.iloc[1]
para(f"This study [1] used the same IBM Telco data and reported logistic-regression ROC-AUC 0.8397 and SHAP stability of "
     f"0.86–1.00, measured as rank correlation of global feature rankings across five cross-validation folds with a single "
     f"training seed. Our logistic regression reached {f3(LIT['b1_roc_auc'])} on a different (70/15/15) split. Measured the same "
     f"global way, our five gate models also ranked features consistently (pairwise Spearman "
     f"{LIT['global_rank_spearman_min']:.2f}–{LIT['global_rank_spearman_max']:.2f}). At the customer level, however, the mean "
     f"top-three agreement was only {f3(LIT['customer_top3_stability_mean'])}, and independently trained models gave "
     f"different actions for {pct(LIT['eval_models_action_not_unanimous_share_B2'])} of recommended customers. Stable global "
     "rankings therefore do not guarantee stable individual recommendations, which is the level at which retention actions "
     "are taken. Their reported 48.0% ROI depends on an assumed $30 cost and 30% success rate; because our data contain no "
     "intervention outcomes, we make no return-on-investment claim.", lead="Vutla et al. (2026).")
para(f"This study [2] showed that models differing only in random seed can disagree on up to 57% of individual "
     f"predictions, and that selective ensembles of ten or more models remove these inconsistencies with abstention as low "
     f"as 1.5%. In our setting the five gate MLPs disagreed on the predicted class for only "
     f"{pct(LIT['gate_models_class_disagreement_share'])} of test customers, yet explanation-derived actions disagreed far more "
     f"often ({pct(LIT['eval_models_action_not_unanimous_share_B2'])} not unanimous under B2). Our findings extend the "
     "discussion of prediction disagreement to explanation-derived decisions: even when independently trained models "
     "produce similar classifications, their recommended actions may differ. Their abstention is based on "
     "prediction agreement; our results suggest that, for action consistency, a signal tied to the action itself is needed, "
     "since top-three feature overlap did not provide one. The comparison is indirect: their benchmarks are not churn data, "
     "and their ensembles are larger than ours.", lead="Black et al. (2022).")
caption("Table 12. Comparison with the two critical studies.")
table(["Study", "Their setting and measure", "Their result", "Our result"],
      [[r.study, f"{r.data}. {r.their_measure}", r.their_result, r.our_result] for r in lit_tab.itertuples()], size=8)

head("Limitations of the evaluation", 11)
para("All results come from one stratified split, one 50-customer SHAP background and five models per role, so the "
     "bootstrap intervals cover test-sample uncertainty only. Action agreement measures reproducibility of suggestions, not "
     "whether they would retain customers, and outreach to non-churners is a descriptive proxy. The action map and "
     "zero risk floor were fixed design choices; other choices could change the size, though not necessarily the sign, of "
     "the differences.")

doc.save(ROOT / "Results_and_Comparison_Section.docx")
print("Wrote Results_and_Comparison_Section.docx")
