# Training and inference guide

## Defaults and parameter overrides

`configs/<model>.json` contains `model`, `model_params`, training budget, batch size, stopping settings, training seed, split seed and dataset choice. A custom JSON file merges over the selected configuration; `model_params` is merged by key. Explicit command-line arguments take precedence. For example, save `{"model_params":{"hidden_dim":256,"n_blocks":4}}` as a JSON file and pass `--config path/to/config.json` for `fm_cfg`. This creates a new configuration rather than reproducing the selected paper architecture.

Supported training flags: `--model`, `--config`, `--epochs`, `--batch-size`, `--lr`, `--device`, `--threads`, `--seed`, `--split-seed`, `--data-dir`, `--dataset`, `--output-dir`, `--patience`, `--min-epochs`, `--resume`, `--stop-after`, `--max-batches`. Run `python scripts/train.py --help` for details. For cWGAN-GP, explicit `--lr` sets both learning rates; separate rates can be set as `lr_g` and `lr_d` in `model_params`.

Default generators use merged data, 300 epochs, batch size 256, a training seed and NumPy split seed of 42, and selected hyperparameters. Non-GAN generators use validation-loss early stopping after the minimum epoch budget; cWGAN-GP retains its internal alternating optimizer updates and has no early stopping. TeM uses CFD data, 300 epochs, batch size 32, learning rate 0.0001, a training seed and Torch split seed of 15, and no early stopping. Standard FM retains its original ablation settings. Actual seeds and overrides are saved in each run.

A generator can train without a surrogate checkpoint. Surrogate evaluation and downstream-MSE HPO require a trained TeM. The predictor uses an 80/10/10 split; the second subset is used for validation by the supplied training implementation, and the final subset is reserved. The separate robustness experiment uses its original fixed partition/validation assignment, recorded in its source.

## Outputs and continuation

When no output directory is supplied, a timestamped directory is created under `runs/<model>/`. With `--model all --output-dir runs/comparison`, each main generator gets its own subdirectory. Existing nonempty output directories are rejected for new runs.

| File | Purpose |
|---|---|
| `config.json` | Effective configuration |
| `data_manifest.json` | Hashes of training/reference arrays |
| `split_indices.npz` | Zero-based train/validation/test indices |
| `history.json` | Training and validation losses per epoch |
| `best.pt` | Best validation checkpoint, architecture and normalization |
| `last.pt` | Full epoch-boundary continuation state |

To intentionally split a 300-epoch run into two invocations:

```bash
python scripts/train.py --model fm_cfg --epochs 300 --stop-after 10 --output-dir runs/continued
python scripts/train.py --resume runs/continued
```

`--stop-after` stops this invocation without changing the planned cosine schedule. Continuation restores weights, optimizer(s), scheduler, best loss, early-stop counters and Python/NumPy/Torch RNG state, including the GAN update counter. It verifies the input hashes. It uses the saved configuration; use a new run for changed settings. Device and data location may change, but exact floating-point continuation is tested on the same CPU environment, not guaranteed across devices. Resume only checkpoints you trust, because optimizer/RNG state uses PyTorch serialization.

## Target input and inference

```bash
python scripts/inference.py --run-dir runs/fm_cfg --predictor-run runs/tem --n-cond 100 --n-samples 10
python scripts/inference.py --run-dir runs/fm_cfg --targets targets.csv --n-samples 20 --output-dir outputs/my_candidates
```

Targets are a headerless numeric CSV or `.npy`, shape `(N,3)`, in physical units and the order **flow, efficiency, pressure ratio**. Without `--targets`, the held-out labels from the run's saved dataset split are used. `--n-cond` applies to this automatic selection; an explicit target file uses all its rows. A single target can be passed as shape `(1,3)`. `best.pt` determines architecture and scaling. Missing checkpoints generate an instruction rather than starting a training job.

Generation outputs include normalized and physical geometry, repeated target performance, a combined CSV and provenance. Optional surrogate evaluation adds predicted performance and per-variable metrics. R² and range-normalized RMSE are `null` when targets have zero variance/range. CFD verification is a separate engineering workflow.

## Optional hyperparameter search

```bash
python scripts/optuna_hpo.py --model fm_cfg --n_trials 50 --retrain --predictor-run runs/tem
python models/performance_predictor/optuna_hpo.py --n-trials 100 --epochs 300
```

The generator search uses the original downstream surrogate objective. `--n_trials` follows the original script's completed-trial target; pruned trials can increase the number of attempted trials. The default target is 40; the command above explicitly requests 50. Archived records retain their original trial counts. The original HPO entrypoints save under the model's local `results/` and `checkpoints/`; they do not overwrite `paper_results/`. Legacy inference from these weights is available with `python scripts/inference.py --model fm_cfg` and the shared predictor checkpoint configuration.

Predictor architecture HPO is a separate portable entrypoint. Its chosen architecture and weights may differ from the fixed paper predictor; do not relabel them as the fixed architecture. The unified training configuration can use a deliberately chosen architecture for a new experiment.

## Troubleshooting and software tests

- CUDA unavailable: choose `--device cpu`, or install the PyTorch CUDA build matching your machine.
- Out of memory: reduce `--batch-size`; a smaller architecture is possible through `--config` but changes the experiment.
- Missing data: run `scripts/check_data.py`; `--data-dir` must contain the documented `.npy` names.
- Output already exists: use a new directory, or `--resume` for an interrupted training run.
- Missing predictor: first train `performance_predictor`, then pass its run directory. Original experimental scripts can use `TURBO_PREDICTOR_CHECKPOINT` pointing to `best.pt`.
- Run `python scripts/smoke_check.py` and `python -m unittest discover -s tests -v`. Short tests use small configurations and a few batches on released data. They validate software, not paper accuracy.
- `--max-batches 1 --epochs 1` is a quick pipeline check; the reduced budget is saved in `config.json`.

Full research training and all historical random-seed experiments are not repeated by the release's short test suite.
