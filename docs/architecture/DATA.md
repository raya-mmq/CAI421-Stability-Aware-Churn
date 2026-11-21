# Shared data pipeline (Student 3)

1. Download Telco-Customer-Churn.csv from the IBM GitHub link in the proposal and put it in `data/raw/`
   (do NOT commit it: dataset license unverified). Expected SHA-256:
   `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91`
2. `pip install -r requirements.txt`
3. `python preprocess.py`   (seed 42, deterministic; the script warns if the SHA-256 differs)

Everyone loads the same splits:

    from preprocess import load_splits
    X_train, X_val, X_test, y_train, y_val, y_test = load_splits()

Files in `data/processed/`: X_/y_ train/val/test (45 features, scaled + one-hot),
idx_* (row indices in the raw CSV), feature_names.json, feature_groups.json
(original feature -> encoded columns; use it to aggregate SHAP values), preprocessing_report.json.

Rules: split BEFORE fitting; scaler and one-hot encoder fitted on train only;
TotalCharges blanks (11 rows, all tenure = 0) set to 0. Do not re-split or re-scale.
