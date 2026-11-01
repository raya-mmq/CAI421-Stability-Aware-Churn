# MLP baseline (Student 5)

Uses Student 3's shared splits only (`load_splits()`); nothing is re-split or re-scaled.

    pip install -r requirements.txt
    python preprocess.py        # needs data/raw/Telco-Customer-Churn.csv (see DATA_README.md)
    python gradient_check.py    # checks backprop against finite differences -> results/gradient_check.json
    python train_mlp.py         # ~30 s on CPU -> results/ and figures/

Files
- `mlp.py`            NumPy MLP: forward pass, backpropagation, Adam, early stopping
- `gradient_check.py` numerical check of the backprop equations (relative error ~1e-8)
- `train_mlp.py`      24-config grid search on VALIDATION ROC-AUC, frozen config tested once on TEST,
                      re-trained with seeds 42-46 for mean +/- std
- `results/mlp_results.json`  all metrics (seed 42 = reported model, plus 5-seed summary)
- `results/mlp_tuning.csv`    every tuning run
- `results/mlp_test_predictions_seed42.csv`, `results/mlp_seed42_weights.npz`
- `figures/` learning curve, confusion matrix, ROC curve, and `mlp_results_panel.png` (Fig. 2 in the paper), 300 dpi

Fixed settings: seed 42, threshold 0.5, max 300 epochs, early-stopping patience 20 on validation BCE
(best epoch restored). Re-running gives identical metrics (only the timing fields change).
