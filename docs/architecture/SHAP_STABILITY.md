# CAI 421 — SHAP Stability component (Student 2)

## What this package does

- Reuses the team's fixed preprocessed 45-feature train/validation/test partitions (seed 42).
- Trains five NumPy MLP gate models (seeds 1–5) with the Checkpoint 2 configuration
  (45–32–1, Adam lr 0.003, batch 128, L2 1e-4, max 300 epochs, patience 20).
- Computes Kernel SHAP on the churn probability with `kernel_shap_numpy.py`
  (same value function as `shap.KernelExplainer`; ~6,000 coalitions per customer).
- Uses one shared 50-row background sampled from the training split (seed 2026).
- Aggregates signed one-hot SHAP values to the 19 original features (`feature_groups.json`).
- Takes each model's top-3 original features (by |SHAP|) and computes the per-customer
  stability S(x) = mean pairwise Jaccard over the 10 gate-model pairs.
- Re-explains 200 customers with an independent coalition design to measure the
  estimator's own noise (reported in the summary JSON).

## Run

```bash
pip install -r requirements.txt
python kernel_shap_numpy.py                     # self-test against brute-force Shapley values
python run_shap_stability.py --split val        # ~15 min on one CPU core
python run_shap_stability.py --split test       # can run in parallel with the line above
python make_shap_report.py                      # figures + SHAP_Stability_Methods_Results.docx
```

The processed files must be in `data/processed/` (produced by `preprocess.py`, see
`DATA_README.md`). Test labels are never read by these scripts.

For the five **evaluation** models (Part 3/4), reuse the same script without touching
the gate outputs:

```bash
python run_shap_stability.py --split test --seed-list 101 102 103 104 105 --out-dir results/shap_eval --noise-check 0
```

## Main outputs (`results/shap_stability/`)

| File | Content |
|---|---|
| `customer_shap_stability_<split>.csv` | one row per customer: `shap_stability`, `mean_gate_probability` (p̄(x) for the gate), each seed's probability, each seed's top-3 features |
| `aggregated_shap_<split>_seed_<s>.csv` / `.npy` | signed 19-feature SHAP matrix (needed for the "highest positive actionable driver" rule); column order in `aggregated_feature_order.json` |
| `pairwise_jaccard_<split>.csv` | the 10 pairwise Jaccard values per customer |
| `top3_features_<split>_seed_<s>.csv` | per-model top-3 lists |
| `global_mean_abs_shap_<split>.csv` | descriptive global importance |
| `gate_model_training_summary_<split>.csv` | best epoch, validation BCE and ROC-AUC per seed |
| `gate_mlp_seed_<s>_weights.npz`, `training_history_seed_<s>.csv` | trained gate models |
| `shap_stability_summary_<split>.json`, `shap_stability_summary.json` | settings, stability statistics, noise check, efficiency check |


## Results (current run)

| Split | n | Mean S (SD) | Median | IQR | S ≥ 0.5 | All 5 identical top-3 | Same top-1 |
|---|---|---|---|---|---|---|---|
| Validation | 1,056 | 0.686 (0.172) | 0.65 | 0.60–0.80 | 85.2% | 15.2% | 63.3% |
| Test | 1,057 | 0.693 (0.170) | 0.65 | 0.60–0.80 | 86.9% | 16.5% | 64.4% |

- Estimator noise check (same model, new coalition design, 200 customers): identical top-3 for 96.5% (val) and 98.0% (test).
- Gate models: validation ROC-AUC 0.844–0.848.
- Top test features by mean |SHAP|: tenure, Contract, InternetService, OnlineSecurity, TechSupport (same top five on validation).

S(x) measures reproducibility of explanations across retrained models. It is not evidence that a feature causes churn or that an intervention works.

## Data and the public repository

This package contains the processed customer-level splits in `data/processed/` **for team use only**.
`.gitignore` excludes `data/` and the per-customer result files, because the paper states that raw and
processed customer data are not redistributed. Do not commit them to GitHub.
