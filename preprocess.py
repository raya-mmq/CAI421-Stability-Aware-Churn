"""
CAI 421 - Checkpoint 2 - Shared preprocessing pipeline (IBM Telco Customer Churn)

Run once:   python preprocess.py
Every team member then loads the SAME splits:

    from preprocess import load_splits
    X_train, X_val, X_test, y_train, y_val, y_test = load_splits()

Pipeline order (no data leakage):
  1. Load raw CSV, record SHA-256
  2. Clean: drop customerID, TotalCharges -> numeric, blanks -> 0, Churn -> 1/0
  3. Stratified 70/15/15 split (seed 42)           <- BEFORE any fitting
  4. Fit scaler + one-hot encoder on TRAIN ONLY, transform val/test
  5. Save arrays, feature names, feature groups (for SHAP aggregation), report
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
ROOT = Path(__file__).resolve().parent
RAW_PATH = ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
OUT_DIR = ROOT / "data" / "processed"

TARGET = "Churn"
ID_COL = "customerID"
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]   # scaled
BINARY_NUM_COLS = ["SeniorCitizen"]                           # already 0/1, kept as is
EXPECTED_SHA256 = "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 1. Load + provenance -------------------------------------------------
    sha = sha256_of(RAW_PATH)
    if sha != EXPECTED_SHA256:
        print(f"WARNING: SHA-256 differs from the team's reference file!\n  got      {sha}\n  expected {EXPECTED_SHA256}")
    raw = pd.read_csv(RAW_PATH)
    report = {
        "source_file": RAW_PATH.name,
        "source_url": "https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv",
        "sha256": sha,
        "raw_shape": list(raw.shape),
        "seed": SEED,
    }

    # ---- 2. Clean --------------------------------------------------------------
    df = raw.drop(columns=[ID_COL]).copy()
    tc_numeric = pd.to_numeric(df["TotalCharges"], errors="coerce")
    blank_mask = tc_numeric.isna()
    report["totalcharges_blank_rows"] = int(blank_mask.sum())
    report["totalcharges_blank_tenure_values"] = sorted(df.loc[blank_mask, "tenure"].unique().tolist())
    # All blank rows have tenure == 0 (customers not yet billed) -> TotalCharges = 0.
    # This is a deterministic rule, so it cannot leak information across splits.
    df["TotalCharges"] = tc_numeric.fillna(0.0)
    df[TARGET] = (df[TARGET] == "Yes").astype(int)
    report["duplicate_customer_ids"] = int(raw[ID_COL].duplicated().sum())
    report["missing_values_after_cleaning"] = int(df.isna().sum().sum())

    y = df[TARGET]
    X = df.drop(columns=[TARGET])
    cat_cols = [c for c in X.columns if c not in NUMERIC_COLS + BINARY_NUM_COLS]
    report["n_numeric_scaled"] = len(NUMERIC_COLS)
    report["n_binary_numeric"] = len(BINARY_NUM_COLS)
    report["n_categorical"] = len(cat_cols)
    report["categorical_cardinality"] = {c: int(X[c].nunique()) for c in cat_cols}

    # ---- 3. Stratified 70/15/15 split (before any fitting) ---------------------
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=SEED)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED)

    # raw descriptive stats of numeric columns (train only, before scaling)
    report["train_numeric_stats_raw"] = X_train[NUMERIC_COLS].describe().loc[
        ["mean", "std", "min", "50%", "max"]].round(3).to_dict()

    # ---- 4. Fit on TRAIN only --------------------------------------------------
    scaler = StandardScaler().fit(X_train[NUMERIC_COLS])
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(X_train[cat_cols])

    def transform(part: pd.DataFrame) -> pd.DataFrame:
        num = pd.DataFrame(scaler.transform(part[NUMERIC_COLS]),
                           columns=NUMERIC_COLS, index=part.index)
        binary = part[BINARY_NUM_COLS].astype(float)
        cat = pd.DataFrame(ohe.transform(part[cat_cols]),
                           columns=ohe.get_feature_names_out(cat_cols), index=part.index)
        return pd.concat([num, binary, cat], axis=1).astype("float32")

    Xtr, Xva, Xte = transform(X_train), transform(X_val), transform(X_test)
    feature_names = list(Xtr.columns)

    # map each ORIGINAL feature -> its encoded columns (needed to aggregate SHAP later)
    feature_groups = {c: [c] for c in NUMERIC_COLS + BINARY_NUM_COLS}
    for c in cat_cols:
        feature_groups[c] = [f for f in feature_names if f.startswith(c + "_")]
    assert sum(len(v) for v in feature_groups.values()) == len(feature_names)

    # ---- 5. Save ---------------------------------------------------------------
    for name, Xp, yp in [("train", Xtr, y_train), ("val", Xva, y_val), ("test", Xte, y_test)]:
        Xp.to_csv(OUT_DIR / f"X_{name}.csv", index=False)
        yp.to_csv(OUT_DIR / f"y_{name}.csv", index=False)
        # original row indices in the raw CSV -> anyone can verify the split is identical
        pd.Series(Xp.index, name="raw_row_index").to_csv(OUT_DIR / f"idx_{name}.csv", index=False)
    json.dump(feature_names, open(OUT_DIR / "feature_names.json", "w"), indent=2)
    json.dump(feature_groups, open(OUT_DIR / "feature_groups.json", "w"), indent=2)

    n_total = len(df)
    report["split"] = {}
    for name, Xp, yp in [("train", Xtr, y_train), ("val", Xva, y_val), ("test", Xte, y_test)]:
        report["split"][name] = {
            "n": int(len(yp)),
            "pct_of_total": round(100 * len(yp) / n_total, 2),
            "n_churn": int(yp.sum()),
            "n_no_churn": int((1 - yp).sum()),
            "churn_rate_pct": round(100 * yp.mean(), 2),
        }
    report["overall"] = {
        "n": n_total,
        "n_churn": int(y.sum()),
        "n_no_churn": int((1 - y).sum()),
        "churn_rate_pct": round(100 * y.mean(), 2),
    }
    report["n_features_after_encoding"] = len(feature_names)
    report["scaler_mean"] = dict(zip(NUMERIC_COLS, scaler.mean_.round(4).tolist()))
    report["scaler_std"] = dict(zip(NUMERIC_COLS, scaler.scale_.round(4).tolist()))
    report["split_overlap_check"] = {
        "train_val": len(set(Xtr.index) & set(Xva.index)),
        "train_test": len(set(Xtr.index) & set(Xte.index)),
        "val_test": len(set(Xva.index) & set(Xte.index)),
    }
    json.dump(report, open(OUT_DIR / "preprocessing_report.json", "w"), indent=2)

    print(json.dumps(report, indent=2))


def load_splits(processed_dir: Path = OUT_DIR):
    """Load the shared splits. Returns X_train, X_val, X_test, y_train, y_val, y_test (numpy)."""
    out = []
    for kind in ("X", "y"):
        for part in ("train", "val", "test"):
            a = pd.read_csv(Path(processed_dir) / f"{kind}_{part}.csv")
            out.append(a.values.astype("float32") if kind == "X" else a.values.ravel().astype("int64"))
    return tuple(out)


if __name__ == "__main__":
    sys.exit(main())
