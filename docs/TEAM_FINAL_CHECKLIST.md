# CAI 421 — final integration checklist (Student 5)

- [ ] Merge the final-stage ZIP into the existing Checkpoint 2 GitHub repository; retain baseline scripts and project diagrams.
- [ ] Main paper Section 4 keeps per-example `δ = p-y`; if mini-batch gradients are added, use batch-scaled `δ̃ = (p-y)/B`.
- [ ] If initialisation is described, use normal variance `1/n_in` (LeCun-style) for the output layer, not Glorot; do not change trained weights.
- [ ] (Optional) State validation BCE used for early stopping excludes the L2 term.
- [ ] Fix the black-box tau subscripts in paper Tables 2–3 and the orphaned Section 6 heading.
- [ ] Course appendix PDF: footer "Course Appendix", remove near-empty page 2, list the fixed `make_shap_report.py` in Appendix E.
- [ ] Appendix letters match everywhere: proof = Appendix A, AI disclosure = Appendix C, reproduction = Appendix D.
- [ ] Every team member confirms the general AI-use statement in the course appendix matches her own use.
- [ ] Do not publish the raw IBM CSV, processed per-customer data, or protected outputs.
- [ ] After merging, run the baseline scripts and `run_all.sh` once on the integrated repository (the final-stage pipeline alone was already re-run on 19 Nov 2026, see `docs/REPRODUCTION_CHECK.md`).
- [ ] Confirm paper figures, equations, references, tables and target venue format.
- [ ] Choose a source-code licence only with agreement from the team.

## Source-of-truth files

Existing Checkpoint 2 baseline: https://github.com/raya-mmq/CAI421-Stability-Aware-Churn

Final-stage scripts in this ZIP: `run_shap_stability.py`, `abstention_gate.py`, `evaluate_results.py`, etc.

This checklist does not claim that the remote GitHub repository has been updated.
