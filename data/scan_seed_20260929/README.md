# User-supplied real scan development seed

Three user-provided PDFs / ten physical pages; explicit public-sharing permission recorded
verbatim and scoped by document ID + SHA256 in draft_labels.json. Raw PDFs are provided in
the conversation dataset bundle, not committed into this Git repository.

28 selected visual-label drafts and 12 drafted QA cases. No full-page transcripts, no
independent human verification and no trained neural model. The schema validator checks
structure and references, not whether the proposed text is true. Keep training_eligible=false.

All samples are development_seed. The DT391 and Thach Bich assignment documents share a
template leakage group; do not split similar pages/crops into independent train/test sets.
The handwritten margin note remains uncertain. Signature appearance and content date are
separate fields; signature validity is not inferred. See docs/10_REAL_SCAN_SEED.md.
