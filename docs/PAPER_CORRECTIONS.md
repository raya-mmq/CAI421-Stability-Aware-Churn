# Main-paper wording corrections (apply to the original PDF source)

The revised Word course appendix contains paste-ready text in Section E. No previously supplied conference PDF was silently modified.
Section references are to the final paper (`CAI421_FINAL_SUBMISSION.pdf`), whose Method is Section 4 without subsections.

1. Section 4, paragraph 2: the paper already states the per-example `delta(L) = p - y`. If mini-batch gradients are added, write the batch-scaled error as `delta_tilde = (p-y)/B` and use it in the gradients.
2. Section 4, paragraph 1 (optional): the paper does not describe initialisation. If added: He for hidden layers, normal with variance `1/n_in` (LeCun-style) for the output layer, not Glorot normal.
3. Section 4, paragraph 1 (optional): state that validation BCE for early stopping excludes L2 regularization.
4. Tables 2 and 3: the tau subscripts in the column headers print as black boxes; write `tau_p` / `tau_s` or export with embedded fonts.
5. Section 6 heading sits alone at the bottom of page 6; keep it with its first paragraph.
6. Course appendix PDF: footer should read "Course Appendix", not "Final Research Paper"; remove the near-empty page 2; Appendix E must list the fixed `make_shap_report.py` (with `OLD_CMP`).
7. Keep the general AI-use statement from the course appendix, confirmed by every member.
8. Preserve the reported negative SHAP stability gate finding.
