import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parent
PROC = ROOT / 'data' / 'processed'
OUT = ROOT / 'data' / 'gate_inputs'
ACTIONS = ['CONTRACT_OFFER', 'PLAN_REVIEW', 'ADDON_SUPPORT_BUNDLE', 'STREAMING_OFFER', 'BILLING_REVIEW']
FEATURE_TO_ACTION = {'Contract': 'CONTRACT_OFFER', 'MonthlyCharges': 'PLAN_REVIEW', 'InternetService': 'PLAN_REVIEW', 'PhoneService': 'PLAN_REVIEW', 'MultipleLines': 'PLAN_REVIEW', 'OnlineSecurity': 'ADDON_SUPPORT_BUNDLE', 'OnlineBackup': 'ADDON_SUPPORT_BUNDLE', 'DeviceProtection': 'ADDON_SUPPORT_BUNDLE', 'TechSupport': 'ADDON_SUPPORT_BUNDLE', 'StreamingTV': 'STREAMING_OFFER', 'StreamingMovies': 'STREAMING_OFFER', 'PaperlessBilling': 'BILLING_REVIEW', 'PaymentMethod': 'BILLING_REVIEW'}

def actions_from_shap(phi, order):
    cols = [j for j, f in enumerate(order) if f in FEATURE_TO_ACTION]
    sub = phi[:, cols]
    best = sub.argmax(axis=1)
    best_val = sub[np.arange(len(sub)), best]
    ids = np.array([ACTIONS.index(FEATURE_TO_ACTION[order[cols[b]]]) for b in best])
    return np.where(best_val > 0, ids, -1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--s2_dir', required=True)
    ap.add_argument('--eval_dir', default=None, help='folder with aggregated_shap_test_seed_101..105.npy')
    a = ap.parse_args()
    r = Path(a.s2_dir) / 'results' / 'shap_stability'
    order = json.load(open(r / 'aggregated_feature_order.json'))
    OUT.mkdir(parents=True, exist_ok=True)
    for split in ('val', 'test'):
        df = pd.read_csv(r / f'customer_shap_stability_{split}.csv')
        idx = pd.read_csv(PROC / f'idx_{split}.csv')['raw_row_index'].to_numpy()
        assert (df['raw_row_index'].to_numpy() == idx).all(), f'{split}: row order differs from the shared split'
        probs = df[[f'seed_{k}_probability' for k in range(1, 6)]].to_numpy()
        assert np.allclose(probs.mean(axis=1), df['mean_gate_probability'], atol=1e-05)
        phi = np.mean([np.load(r / f'aggregated_shap_{split}_seed_{k}.npy') for k in range(1, 6)], axis=0)
        assert phi.shape == (len(df), 19)
        act = actions_from_shap(phi, order)
        np.save(OUT / f'gate_probs_{split}.npy', probs)
        np.save(OUT / f'stability_{split}.npy', df['shap_stability'].to_numpy())
        np.save(OUT / f'gate_action_{split}.npy', act)
        share = (act >= 0).mean()
        print(f'{split}: n={len(df)}  actionable share={share:.3f}  actions={np.bincount(act[act >= 0], minlength=len(ACTIONS)).tolist()}')
    if a.eval_dir:
        e = Path(a.eval_dir)
        cols = [actions_from_shap(np.load(e / f'aggregated_shap_test_seed_{s}.npy'), order) for s in range(101, 106)]
        E = np.stack(cols, axis=1)
        np.save(OUT / 'eval_actions_test.npy', E)
        print(f'eval actions saved: {E.shape}, share with action per model = {(E >= 0).mean(axis=0).round(3).tolist()}')
    else:
        print("NOTE: no --eval_dir given -> eval_actions_test.npy NOT created (needed for 'evaluate').")
    json.dump({'actions': ACTIONS, 'feature_to_action': FEATURE_TO_ACTION, 'feature_order': order}, open(OUT / 'action_map.json', 'w'), indent=2)
if __name__ == '__main__':
    main()
