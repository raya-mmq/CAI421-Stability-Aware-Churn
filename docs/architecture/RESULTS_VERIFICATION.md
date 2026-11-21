# Verification report — revision 2

Inputs: Student 2 `SHAP_Stability_Student2_FINAL.zip` and Student 3 `abstention_gate.zip`, as uploaded.

| Check | Result |
|---|---|
| Python syntax (`py_compile`: evaluate_results.py, make_results_section.py, mlp.py) | Passed |
| Full run of evaluate_results.py + make_results_section.py from a clean folder | Passed |
| Input check, validation: 1,056 rows, order matches `idx_val.csv`, stability and gate probabilities identical (max diff 0) | Passed; mean stability 0.68589 |
| Input check, test: 1,057 rows, order matches `idx_test.csv`, stability and gate probabilities identical (max diff 0) | Passed; mean stability 0.69304 |
| Input check catches bad inputs (on altered copies): one changed value, two swapped rows, one missing row | All three stopped with a clear `INPUT CHECK FAILED` message (exit code 1) |
| Student 3's reported agreement and recommendation counts reproduced from the saved inputs (existing check) | Passed |
| `results/results_summary.json` (all metrics, bootstrap CIs, thresholds, diagnostics) | Byte-identical to the previous version |
| All 7 CSV tables | Byte-identical to the previous version |
| Fig. 4 (`fig_gate_minus_b3.png`) | Pixel-identical |
| Figs. 5 and 6 | Only the relabelled text changed (axis label of Fig. 5 right panel; legend of Fig. 6). Plot areas are pixel-identical |
| Word document vs code | Every number in the document is identical to the previous version. Tables 7, 9 and 10 were checked cell-by-cell against the CSV outputs and match. Only the 7 intended text blocks changed |
| Table 7 layout | Rendered to PDF; confusion-matrix values each fit on one line |
| Students 2 and 3 files | Unchanged (compared with fresh copies of their zips) |

**Did any results change?** No. All numbers, confidence intervals, thresholds, figures (data) and the negative conclusion are the same.

**Not verified:** The Word file was checked by rendering in LibreOffice, not in Microsoft Word. Column widths can look slightly different in Word.
