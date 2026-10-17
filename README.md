# Turbomachinery Inverse Design with Flow Matching and Classifier-Free Guidance

Research code accompanying **A generative framework for turbomachinery aerodynamic inverse design based on flow matching with classifier-free guidance**.

The framework generates 21-dimensional compressor rotor geometry parameters conditioned on three aerodynamic performance targets. A Transformer surrogate predicts performance, and a common evaluation interface supports generative-model comparison and hyperparameter optimization.

## Models and source layout

| Paper / implementation | Directory | Role |
|---|---|---|
| cFM-CFG | `models/fm_cfg` | Conditional flow matching with classifier-free guidance |
| cAR | `models/ar` | Autoregressive Transformer |
| cVAE | `models/cvae` | Conditional variational autoencoder |
| cWGAN-GP | `models/cgan` | Conditional Wasserstein GAN with gradient penalty |
| cDDPM | `models/ddpm` | Conditional denoising diffusion model |
| cINN | `models/cinn` | Conditional affine-coupling normalizing flow |
| FM without CFG | `models/fm` | Ablation implementation |
| DDPM with CFG | `models/ddpm_cfg` | Ablation implementation |
| PA-FM-CFG | `models/pa_fm_cfg` | Performance-aware extension |
| Transformer predictor | `models/performance_predictor` | Geometry-to-performance surrogate |

`shared/` provides data loading, normalization and evaluation. `scripts/` provides Optuna optimization, inference and plotting. `experiments/` contains guidance ablations, generation trajectories, computational-cost measurement, repeated-seed robustness and Monte Carlo dropout inference.

## Installation

Reference environment: Python 3.9.18, PyTorch 2.3.0 (CUDA 11.8 in the original environment). Install a PyTorch build suitable for your hardware, then run from the repository root:

```bash
python -m pip install -r requirements.txt
python scripts/smoke_check.py
```

The smoke check uses small random tensors to check software interfaces. It does not generate research results or require research data. Full training benefits from a CUDA-capable GPU.

## Data and pretrained weights

Research datasets and trained weights are not bundled in this source repository. Obtain the research inputs from the corresponding authors under the study's data-access arrangements. See [data/README.md](data/README.md) for filenames, shapes, variable ordering and normalization.

Place the surrogate checkpoint at:

```text
models/performance_predictor/checkpoints/best_model_in_testdataset.pth
```

The six main generators use `models/<model>/checkpoints/<model>_hpo.pth` and the accompanying `models/<model>/results/hpo_best_<model>.json`. Included JSON files preserve the available local run records; their trial counts and settings are not overwritten to match a different experimental budget.

## Workflow

### 1. Train the performance predictor

```bash
python models/performance_predictor/train.py --epochs 300
```

This saves checkpoints in the location expected by the shared evaluator. Data paths, output directory, batch size and split seed can be set through `--help`. The historical `best_model_in_testdataset.pth` filename is retained for compatibility; selection uses the validation loader in the training script.

Predictor architecture search is available separately:

```bash
python models/performance_predictor/optuna_hpo.py --n-trials 100 --epochs 300
```

This portable entrypoint uses the original predictor search space and saves the selected architecture alongside its weights. Its output is for the searched architecture; the shared evaluator expects the fixed predictor architecture described in the paper, so do not replace that evaluator's checkpoint with weights from a different architecture.

### 2. Optimize and train generative models

```bash
python scripts/optuna_hpo.py --model fm_cfg --n_trials 50 --retrain
python scripts/optuna_hpo.py --model all --n_trials 50 --retrain
```

`all` covers the six main comparison models. The existing command default is 40 trials; the commands above explicitly request a 50-trial budget. HPO evaluates generated geometries through the pretrained surrogate and therefore requires its checkpoint. Each model's `train.py` also provides a direct training entrypoint; most of these scripts use editable settings at the end of the file rather than command-line arguments.

For two GPUs, `python scripts/run_hpo_parallel.py --n_trials 50 --retrain` assigns the six models to devices 0 and 1. Adjust `GPU_GROUPS` for your hardware.

### 3. Generate and evaluate candidates

```bash
python scripts/inference.py --model fm_cfg --n_cond 100 --n_samples 10
python scripts/inference.py --model fm_cfg ar cvae cgan ddpm cinn
```

The script loads each model's HPO settings and checkpoint, generates candidates for held-out conditions, and writes geometry and surrogate-predicted performance arrays plus CSV/JSON summaries. If a generator checkpoint is missing, the original script starts retraining with the saved settings. Surrogate predictions are distinct from subsequent CFD validation.

### 4. Additional analyses

| Analysis | Entry point |
|---|---|
| Guidance scale | `experiments/ablation/sweep_guidance_scale.py` |
| Unconditional training probability | `experiments/ablation/sweep_p_uncond.py` |
| Sampling steps | `experiments/ablation/sweep_sampling_steps.py` |
| Generation trajectories | `experiments/trajectory/generate_trajectory.py` |
| Computational cost | `experiments/computational_cost/run_benchmark.py` |
| Repeated training seeds | `experiments/robustness/robustness_test.py` |
| MC dropout | `experiments/uncertainty/mc_dropout_inference.py` |

Analysis scripts retain their experiment-specific settings and require the relevant data, configurations and trained checkpoints. Configure those settings before running long experiments. CFD solver cases and solver automation are outside this machine-learning source package.

## Implementation and provenance

See [docs/implementation.md](docs/implementation.md) for data splits, normalization, reproducibility settings and the scope of the portability edits. Model architectures, losses and sampling equations are retained from the supplied project. [docs/source_manifest.json](docs/source_manifest.json) records source and release hashes for the collected files.

Third-party attribution for the Transformer implementation is recorded in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). No additional license grant is made for the authors' code by this packaging step.
