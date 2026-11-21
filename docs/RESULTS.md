# CAI 421 — Results & Comparison (Student 4)

Final evaluation of the frozen abstention gates (Student 3), comparison with the two critical studies
(Vutla et al. 2026; Black et al. 2022), and all tables and figures for the Results section.
Nothing from Students 2–3 is changed: their saved outputs are read as-is, and the script first checks
that Student 3's reported agreement and counts are reproduced exactly from the saved gate inputs.

## Run
```bash
pip install -r requirements.txt
python evaluate_results.py --processed <path>/data/processed \
       --s2 <SHAP_Stability_Student2_Package> --s3 <abstention_gate>
python make_results_section.py        # -> Results_and_Comparison_Section.docx
```
Before evaluating, the script checks that Student 3's gate inputs match Student 2's final outputs
(row counts, customer order against `idx_<split>.csv`, then `np.testing.assert_allclose` on the stability
and gate-probability arrays for validation and test). It stops with a clear message if anything differs and
writes `results/input_verification.json` when the check passes.

Runtime about 1 minute (it refits B1 and the seed-42 MLP with the Checkpoint 2 settings for paired comparisons;
both reproduce Checkpoint 2 Tables 4–6 exactly). Test labels are used for evaluation only.

## Outputs
| File | Paper item |
|---|---|
| `tables/table1_predictive_performance.csv` | Table 7 — B1, MLP seed 42, gate ensemble, single gate MLPs |
| (from `results/results_summary.json`) | Table 8 — frozen validation thresholds |
| `tables/table2_policy_comparison.csv` | Table 9 — B2 / B3 / proposed gate at 50/70/90% |
| `tables/table3_gate_minus_B3_bootstrap.csv` | Table 10 — gate − B3 with 95% CIs |
| `tables/table4*_*.csv` | Table 11 — why stability did not help (diagnostics) |
| `tables/table5_published_comparison.csv` | Table 12 — comparison with [1] and [2] |
| `figures/fig_gate_minus_b3.png` | Fig. 4 |
| `figures/fig_agreement_by_signal.png` | Fig. 5 |
| `figures/fig_coverage_curves.png` | Fig. 6 |
| `Results_and_Comparison_Section.docx` | Sections 7–8 text, tables and figures (numbers generated from the files above) |

Table/figure numbers continue Checkpoint 2; renumber when the paper is merged.

## Main findings
- Gate ensemble ROC-AUC 0.846 ≈ B1 0.845 (difference CI includes 0); B1 has higher F1 at 0.5.
- The stability gate did **not** improve action agreement over confidence-only abstention at any target:
  50% −0.001 [−0.015, +0.013]; 70% −0.034 [−0.048, −0.021]; 90% −0.011 [−0.015, −0.006].
  It also raised outreach to non-churners at all three targets.
- Reason: S(x) does not predict evaluation-model agreement (ρ = −0.09), mean churn probability p̄(x) does (ρ = 0.39), and
  high-stability customers are mostly low-risk. Post-hoc: gate-model *action* agreement predicts it well (ρ = 0.74).
- vs [1]: global feature rankings are stable (Spearman 0.78–0.96, like their 0.86–1.00) but per-customer
  explanations are not (mean 0.693; 30.7% of B2 actions not unanimous).
- vs [2]: predictions disagree for only 7.9% of customers, explanation-based actions far more often.
