# Implementation notes

## Preserved computation

Network architecture files, losses and sampling routines are copied without scientific changes. The shared data loader uses an 80/10/10 split with NumPy seed 42. Min/max scaling uses the full CFD reference arrays, including when the augmented dataset is selected. The performance-predictor training script uses split seed 15 and chooses its best checkpoint using its returned validation loader.

The repeated-seed robustness script holds the data partition fixed at seed 15 and varies the training seed. This experiment measures variability across repeated training runs. MC dropout activates dropout during inference and uses the split definitions in its source script.

The common HPO objective is downstream surrogate-prediction MSE on sampled validation conditions. Training objectives and stopping behavior remain model-specific where required (including the GAN's internally managed optimizers). Search spaces, actual pruning behavior and defaults are defined by executable code in `scripts/optuna_hpo.py`. Supplied HPO JSON records retain the settings and trial counts of the corresponding local runs. To request another trial budget, pass `--n_trials` explicitly. Retraining retains separate training and validation loaders.

## Portability edits

- The predictor trainer accepts relative default paths and optional CLI overrides, and saves in the evaluator's expected `checkpoints/` directory.
- The dual-GPU launcher uses the active Python interpreter and schedules the six models accepted by the HPO command.
- Robustness and MC-dropout source scripts use the packaged predictor and dataset implementations, with repository-relative input/checkpoint paths.
- Predictor HPO is a runnable port of `Baysien_optimization.py`: the search space, split seed, optimizer, schedule and five-epoch validation interval are retained. Trial scores are scoped to each trial, saved best states are copied, and architecture parameters are separated from batch size when constructing the model. These bookkeeping corrections apply to new runs; no historical experiment is regenerated or relabelled. The last epoch is also evaluated to support custom epoch budgets.
- Documentation, dependency pins, an input schema and a data-free smoke check are added.

No raw CFD cases, research arrays, trained weights, manuscript text or review correspondence are included. This source package supports the machine-learning workflows; reproducing trained-model numerical results additionally requires the corresponding data, weights and experiment settings.

## Validation scope

`python scripts/smoke_check.py` checks finite losses, gradients and sample dimensions using small model configurations on CPU, plus a Transformer forward pass. It is a software check, not CFD or scientific validation. Long training runs are not repeated as part of packaging.

The cost-summary script retains numerical table generation and omits the original optional editorial-reply drafting branch.
