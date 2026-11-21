# End-to-end reproduction check (Student 5)

Date: 2026-11-19 · Python 3.13 · numpy 2.5.3, pandas 3.0.5, scikit-learn 1.9.1, scipy 1.18.1, matplotlib 3.11.2 · 2 CPU cores.

Starting point: the shared `data/processed/` splits only (the raw CSV was not re-downloaded, so `preprocess.py`
was not re-run). Every other step in `run_all.sh` was run from scratch in a clean folder.

| Check | Result |
|---|---|
| `tests/gradcheck.py` (backprop vs finite differences, 45–32–1 and 45–16–8–1) | Passed, max relative error 2.0e-9 / 2.2e-9 |
| `kernel_shap_numpy.py` self-test | Passed (exact match with brute-force Shapley, M = 8) |
| `run_shap_stability.py` val and test (gate seeds 1–5) | Same outputs as Student 2 |
| `run_eval_seed.py` 101–105 → `build_gate_inputs.py` | Gate inputs identical to Student 3's |
| `abstention_gate.py select` / `evaluate` | `thresholds.json` and `gate_results.*` identical |
| `evaluate_results.py`, `make_results_section.py` | All tables, `results_summary.json`, input check identical to Student 4's |
| 74 result files compared with teammates' uploads | 70 byte-identical; 4 differ only in runtime or a hand-added `note` field |
| 5 figures | Pixel-identical (byte-identical PNGs) |
| Text of both Word sections | Identical, paragraph by paragraph and cell by cell |
| Total runtime | About 30 minutes (SHAP val and test in parallel, then 5 × ~100 s evaluation models) |

## Fix made during the check

`make_shap_report.py` stopped with `NameError: name 'old_cmp' is not defined`: the variable holding the
superseded 128-coalition run's figures had been removed from the script but was still used in the Methods text.
It is now defined as a constant (`OLD_CMP`) with the values already reported in the section
(76.5%, 0.881, 0.656, 0.664). The generated text is unchanged. No other code was changed.
