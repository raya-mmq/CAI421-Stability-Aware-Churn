# Abstention Gate (B2 / B3 / Confidence + Stability)

This component compares three recommendation policies on the IBM Telco churn data:

- **B2**: recommend whenever a customer has an actionable driver (no abstention).
- **B3**: confidence-only abstention (threshold on mean gate-ensemble churn probability).
- **Proposed gate**: confidence threshold plus an explanation-stability threshold.

Requires Python 3.12+ (see `requirements.txt` for packages).

## Run order

1. Build the gate inputs from the SHAP-stability outputs:
   `python build_gate_inputs.py --s2_dir <SHAP_Stability_Package> --eval_dir <...>/results/shap_eval`
2. Select thresholds on the validation split only (written to `results/gate/thresholds.json` and then frozen):
   `python abstention_gate.py select`
3. Evaluate once on the test split with the frozen thresholds (writes `results/gate/gate_results.csv` and `.json`):
   `python abstention_gate.py evaluate`
4. Optional smoke test on synthetic data (numbers are not meaningful):
   `python abstention_gate.py demo`

The five evaluation models (seeds 101-105) are produced with the SHAP-stability code:
`python run_eval_seed.py 101` ... `python run_eval_seed.py 105`. Place `run_eval_seed.py` in the
SHAP-stability package folder; it uses the same settings as `run_shap_stability.py --split test`,
one seed per run.

## Inputs (`data/gate_inputs/`, generated locally; not redistributed)

These files contain per-customer values derived from the IBM Telco dataset. They are excluded from
version control and can be regenerated with `build_gate_inputs.py`.

| File | Shape | Content |
|---|---|---|
| `gate_probs_{val,test}.npy` | (n, 5) | churn probability of each of the 5 gate models |
| `stability_{val,test}.npy` | (n,) | explanation stability S(x) in [0, 1] |
| `gate_action_{val,test}.npy` | (n,) | suggested action id (int), -1 = no actionable driver |
| `eval_actions_test.npy` | (n, 5) | action suggested by each evaluation model (test only) |
| `action_map.json` | - | action names, feature-to-action map, feature order |

## Design decisions

- **Actionable features** map to predefined, non-causal suggestions (`FEATURE_TO_ACTION` in
  `build_gate_inputs.py`). Not actionable: tenure, TotalCharges, SeniorCitizen, gender, Partner, Dependents.
- **Gate action**: the actionable feature with the highest positive signed SHAP value, averaged over the
  5 gate models. Each evaluation model uses its own SHAP values.
- **Risk eligibility**: `RISK_FLOOR = 0.0` in `abstention_gate.py`, so eligibility is determined by the
  risk threshold tau_p itself.

## Method (paper Sections 6.3 and 6.4)

Coverage targets are 50%, 70% and 90% of all evaluated customers. Thresholds are chosen on validation only:

- **B3**: the largest tau_p that reaches the target coverage.
- **Gate**: the largest tau_s on {0.0, 0.1, ..., 1.0} for which the target is attainable, then the largest
  tau_p that reaches it.

On the test split the frozen thresholds are applied once and the achieved coverage is reported.
Recommendation agreement is the mean pairwise agreement over the 10 evaluation-model pairs among
recommended customers. Differences between gates use a paired bootstrap (2,000 resamples, seed 42).
