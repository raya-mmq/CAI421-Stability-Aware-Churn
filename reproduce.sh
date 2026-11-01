#!/usr/bin/env bash
# Rebuild every Student 4 result from the raw CSV. Run from the repository root.
set -euo pipefail
CSV=data/raw/Telco-Customer-Churn.csv
EXPECTED=16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91
if [ ! -f "$CSV" ]; then
  echo "Missing $CSV - see 'Obtaining the dataset' in README.md"; exit 1
fi
GOT=$(python -c "import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" "$CSV")
[ "$GOT" = "$EXPECTED" ] || { echo "SHA-256 mismatch: $GOT"; exit 1; }
python preprocess.py > /dev/null           # Student 3: shared splits (deterministic)
python train_logistic_regression.py        # Student 4: B1 baseline (checks processed-file hashes)
python compare_models.py                   # Student 4: B1 vs saved MLP seed-42 results
echo "Done. See results/logreg/, results/comparison/, figures/logreg/."
