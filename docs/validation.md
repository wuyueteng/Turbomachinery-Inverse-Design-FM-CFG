# Validation scope

Release checks include Python syntax/imports, dataset shapes and finiteness, exact paired-data provenance, CSV roundtrip and source checksums. The benchmark verification script recomputes six-model regression metrics, k=5 diversity/novelty and distribution distances, compares Table 1 rounding, and verifies archived Figure 2 summaries.

The training suite exercises all eight included network types with small configurations and short budgets. It compares uninterrupted training with epoch-boundary continuation, including both GAN optimizers and its update counter. CLI integration covers predictor training, cFM-CFG training, candidate generation, surrogate evaluation, and clear missing-input errors. The plotting command runs without pretrained weights.

These checks validate the software package and numerical archives. Full research training, all repeated-seed experiments and CFD solutions are not rerun as part of release validation. New stochastic training outputs are not claimed to reproduce the archived realization exactly.

## v1.0.0 check results (17 October 2026)

All listed checks passed in an independent checkout of the staged release. The two workflow tests exercise all eight network types, including exact CPU continuation and the predictor–generator–evaluation CLI sequence. The archive check verified 118 source artifacts; the plotting entrypoint produced 10 figure groups (20 PNG/PDF files). See [the machine-readable validation report](release_validation.json) and [figure previews](../paper_results/previews/README.md).
