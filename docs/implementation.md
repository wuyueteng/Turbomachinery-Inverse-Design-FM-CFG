# Implementation and release notes

The current release includes the six paper generators, standard FM for the CFG ablation, and the Transformer predictor. Core model equations, losses and sampling routines retain the supplied implementation. Source hashes and numeric provenance are recorded in the release manifests.

The unified entrypoint separates command-line settings from network code, uses model-specific selected configurations, and writes independent runs. Predictor min/max scaling uses the complete CFD input arrays; generator scaling uses the same reference even for merged data. The original split algorithms and default seeds are documented in the training guide. Training seeds are explicit and separate from split seeds.

Generator training uses Adam and a cosine schedule, gradient clipping at 1, and validation-loss model selection, with the GAN preserving its internally managed dual optimizers and update schedule. GAN validation computes the generator objective without optimizer updates; early stopping is disabled for it. Predictor loss follows the original average of batch MSE values and uses no gradient clipping or early stopping. Standard FM retains its original standalone ablation settings. The generic entrypoint saves full continuation state at each completed epoch.

The predictor architecture-search port retains the original search space and schedule. It scopes scores per trial, copies best states and separates architecture fields from batch size when reconstructing checkpoints. These bookkeeping fixes apply to new runs; historical results are not regenerated or relabelled. The six-model HPO search keeps the existing objective and trial accounting. Archive files are separate from newly generated HPO output.

Legacy training scripts remain available. The README recommends the unified `scripts/train.py` API and run-based inference. Legacy inference no longer triggers training when a checkpoint is absent. The frozen evaluator can load either the historical state dictionary or a unified predictor `best.pt`.

Paper replotting uses archived numerical results without model inference. The original uncertainty-calibration and panel calculations are retained in the plotting helper. Other replot layouts use portable fonts; chart identifiers and numerical definitions are documented in the figure map.
