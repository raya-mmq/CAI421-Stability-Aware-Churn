#!/usr/bin/env bash
# Full pipeline, in order. About 40 min on 2 CPU cores (SHAP dominates).
# Needs data/raw/Telco-Customer-Churn.csv (see docs/DATA.md) or an existing data/processed/.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d data/processed ]; then python preprocess.py; fi
python tests/gradcheck.py                                   # backprop gradient check (course Appendix A)
python kernel_shap_numpy.py                                 # SHAP self-test

# 1. SHAP stability, gate models (seeds 1-5)
python run_shap_stability.py --split val
python run_shap_stability.py --split test
# 2. Evaluation models (seeds 101-105), test split only
for s in 101 102 103 104 105; do python run_eval_seed.py "$s"; done
# 3. Abstention gate: thresholds on validation, then ONE test evaluation
python build_gate_inputs.py --s2_dir . --eval_dir results/shap_eval
python abstention_gate.py select
python abstention_gate.py evaluate
# 4. Figures, tables and Word sections
python make_shap_report.py
python evaluate_results.py --processed data/processed --s2 . --s3 .
python make_results_section.py
echo "Done. Tables in tables/, figures in figures/, summaries in results/."
