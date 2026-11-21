import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score
ROOT = Path(__file__).resolve().parent
PROC = ROOT / 'data' / 'processed'
SEED = 42
TARGETS = (0.5, 0.7, 0.9)
GRID = np.round(np.arange(0.0, 1.0001, 0.1), 1)
N_BOOT = 2000
RISK_FLOOR = 0.0
EPS = 1e-09

def load_split(split, in_dir):
    in_dir = Path(in_dir)
    d = {'probs': np.load(in_dir / f'gate_probs_{split}.npy'), 'S': np.load(in_dir / f'stability_{split}.npy').astype(float), 'a': np.load(in_dir / f'gate_action_{split}.npy').astype(int), 'y': pd.read_csv(PROC / f'y_{split}.csv').values.ravel().astype(int)}
    n = len(d['y'])
    assert d['probs'].shape == (n, 5), f'gate_probs_{split}: expected ({n}, 5), got {d['probs'].shape}'
    assert d['S'].shape == (n,) and d['a'].shape == (n,), f'{split}: stability/action must have {n} rows'
    assert d['S'].min() >= -EPS and d['S'].max() <= 1 + EPS, 'stability must be in [0, 1]'
    d['pbar'] = d['probs'].mean(axis=1)
    if split == 'test':
        e = np.load(in_dir / 'eval_actions_test.npy').astype(int)
        assert e.shape == (n, 5), f'eval_actions_test: expected ({n}, 5), got {e.shape}'
        d['eval'] = e
    return d

def recommend(pbar, S, a, tau_p, tau_s):
    return (a >= 0) & (pbar >= max(tau_p, RISK_FLOOR) - EPS) & (S >= tau_s - EPS)

def _kth_threshold(pbar, mask, k):
    vals = np.sort(pbar[mask])[::-1]
    return None if len(vals) < k else float(vals[k - 1])

def select_thresholds(pbar, S, a, c):
    k = int(np.ceil(c * len(pbar) - EPS))
    base = (a >= 0) & (pbar >= RISK_FLOOR - EPS)
    tp_b3 = _kth_threshold(pbar, base, k)
    if tp_b3 is None:
        return None
    for ts in GRID[::-1]:
        tp = _kth_threshold(pbar, base & (S >= ts - EPS), k)
        if tp is not None:
            return {'b3_tau_p': tp_b3, 'gate_tau_s': float(ts), 'gate_tau_p': tp}
    return None

def pairwise_agreement(E):
    n, M = E.shape
    agree, pairs = (np.zeros(n), 0)
    for i in range(M):
        for j in range(i + 1, M):
            agree += E[:, i] == E[:, j]
            pairs += 1
    return agree / pairs

def system_metrics(rec, y, agr, unan):
    R = int(rec.sum())
    nan = float('nan')
    return {'coverage': float(rec.mean()), 'abstention': float(1 - rec.mean()), 'n_recommended': R, 'agreement': float(agr[rec].mean()) if R else nan, 'unanimous': float(unan[rec].mean()) if R else nan, 'outreach_to_nonchurn': float((y[rec] == 0).mean()) if R else nan}

def paired_bootstrap(rec_a, rec_b, y, agr, unan, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(y)
    keys = ['coverage', 'agreement', 'unanimous', 'outreach_to_nonchurn']
    diffs = {k: [] for k in keys}
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        ma = system_metrics(rec_a[i], y[i], agr[i], unan[i])
        mb = system_metrics(rec_b[i], y[i], agr[i], unan[i])
        for k in keys:
            diffs[k].append(ma[k] - mb[k])
    out = {}
    for k in keys:
        lo, hi = np.nanpercentile(diffs[k], [2.5, 97.5])
        out[k] = [float(lo), float(hi)]
    return out

def ensemble_metrics(pbar, y):
    return {'roc_auc': float(roc_auc_score(y, pbar)), 'pr_auc': float(average_precision_score(y, pbar)), 'f1_churn_at_0.5': float(f1_score(y, pbar >= 0.5)), 'brier': float(brier_score_loss(y, pbar))}

def stage_select(in_dir, out_dir):
    v = load_split('val', in_dir)
    res = {'risk_floor': RISK_FLOOR, 'grid': GRID.tolist(), 'n_val': int(len(v['y'])), 'max_possible_coverage_val': float((v['a'] >= 0).mean()), 'targets': {}}
    for c in TARGETS:
        t = select_thresholds(v['pbar'], v['S'], v['a'], c)
        if t is None:
            res['targets'][str(c)] = {'feasible': False}
            print(f'target {c:.0%}: INFEASIBLE on validation (max possible {res['max_possible_coverage_val']:.1%})')
            continue
        cov_b3 = recommend(v['pbar'], v['S'], v['a'], t['b3_tau_p'], 0.0).mean()
        cov_g = recommend(v['pbar'], v['S'], v['a'], t['gate_tau_p'], t['gate_tau_s']).mean()
        res['targets'][str(c)] = {'feasible': True, **t, 'val_cov_b3': float(cov_b3), 'val_cov_gate': float(cov_g)}
        print(f'target {c:.0%}: B3 tau_p={t['b3_tau_p']:.4f} (val cov {cov_b3:.3f}) | gate tau_s={t['gate_tau_s']:.1f}, tau_p={t['gate_tau_p']:.4f} (val cov {cov_g:.3f})')
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out_dir / 'thresholds.json', 'w'), indent=2)
    print(f"Thresholds frozen -> {out_dir / 'thresholds.json'}  (do NOT re-run 'select' after looking at test results)")

def stage_evaluate(in_dir, out_dir):
    th_path = out_dir / 'thresholds.json'
    if not th_path.exists():
        sys.exit("thresholds.json not found: run 'select' on validation first.")
    th = json.load(open(th_path))
    t = load_split('test', in_dir)
    y, pbar, S, a = (t['y'], t['pbar'], t['S'], t['a'])
    agr, unan = (pairwise_agreement(t['eval']), (t['eval'] == t['eval'][:, [0]]).all(axis=1))
    rows, report = ([], {'ensemble_metrics_test': ensemble_metrics(pbar, y), 'n_test': int(len(y)), 'targets': {}})
    rec_b2 = recommend(pbar, S, a, 0.0, 0.0)
    b2 = system_metrics(rec_b2, y, agr, unan)
    report['B2'] = b2
    rows.append({'target': '-', 'system': 'B2', **b2})
    for c in TARGETS:
        tc = th['targets'][str(c)]
        if not tc['feasible']:
            report['targets'][str(c)] = {'feasible': False}
            continue
        rec_b3 = recommend(pbar, S, a, tc['b3_tau_p'], 0.0)
        rec_g = recommend(pbar, S, a, tc['gate_tau_p'], tc['gate_tau_s'])
        mb3, mg = (system_metrics(r, y, agr, unan) for r in (rec_b3, rec_g))
        ci = paired_bootstrap(rec_g, rec_b3, y, agr, unan)
        diff = {k: mg[k] - mb3[k] for k in ci}
        report['targets'][str(c)] = {'feasible': True, 'B3': mb3, 'gate': mg, 'diff_gate_minus_B3': diff, 'ci95': ci, 'thresholds': {k: tc[k] for k in ('b3_tau_p', 'gate_tau_s', 'gate_tau_p')}}
        rows.append({'target': c, 'system': 'B3 confidence-only', **mb3})
        rows.append({'target': c, 'system': 'Gate confidence+stability', **mg})
        print(f'\ntarget {c:.0%} | achieved test coverage: B3 {mb3['coverage']:.3f}, gate {mg['coverage']:.3f}')
        for k in ('agreement', 'unanimous', 'outreach_to_nonchurn'):
            print(f'  {k:22s} B3 {mb3[k]:.3f} | gate {mg[k]:.3f} | diff {diff[k]:+.3f}  95% CI [{ci[k][0]:+.3f}, {ci[k][1]:+.3f}]')
    pd.DataFrame(rows).round(4).to_csv(out_dir / 'gate_results.csv', index=False)
    json.dump(report, open(out_dir / 'gate_results.json', 'w'), indent=2)
    print(f'\nSaved {out_dir / 'gate_results.csv'} and gate_results.json')

def make_demo_inputs(in_dir, seed=0):
    rng = np.random.default_rng(seed)
    in_dir.mkdir(parents=True, exist_ok=True)
    for split in ('val', 'test'):
        y = pd.read_csv(PROC / f'y_{split}.csv').values.ravel()
        n = len(y)
        base = np.clip(0.2 + 0.35 * y + rng.normal(0, 0.15, n), 0.01, 0.99)
        np.save(in_dir / f'gate_probs_{split}.npy', np.clip(base[:, None] + rng.normal(0, 0.03, (n, 5)), 0, 1))
        S = np.clip(rng.beta(2, 2, n), 0, 1)
        np.save(in_dir / f'stability_{split}.npy', S)
        act = np.where(rng.random(n) < 0.75, rng.integers(0, 4, n), -1)
        np.save(in_dir / f'gate_action_{split}.npy', act)
        if split == 'test':
            same = rng.random((n, 5)) < 0.5 + 0.5 * S[:, None]
            alt = rng.integers(0, 4, (n, 5))
            np.save(in_dir / 'eval_actions_test.npy', np.where(same, act[:, None], alt))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['select', 'evaluate', 'demo'])
    ap.add_argument('--in_dir', default=str(ROOT / 'data' / 'gate_inputs'))
    ap.add_argument('--out_dir', default=str(ROOT / 'results' / 'gate'))
    a = ap.parse_args()
    in_dir, out_dir = (Path(a.in_dir), Path(a.out_dir))
    if a.stage == 'demo':
        in_dir, out_dir = (ROOT / 'demo_tmp' / 'inputs', ROOT / 'demo_tmp' / 'results')
        print('*** DEMO on SYNTHETIC data: numbers are meaningless, only checks that the code runs ***\n')
        make_demo_inputs(in_dir)
        stage_select(in_dir, out_dir)
        stage_evaluate(in_dir, out_dir)
    elif a.stage == 'select':
        stage_select(in_dir, out_dir)
    else:
        stage_evaluate(in_dir, out_dir)
if __name__ == '__main__':
    main()
