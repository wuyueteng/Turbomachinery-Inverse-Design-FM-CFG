# Turbomachinery inverse design with flow matching and classifier-free guidance

[中文说明](README_zh.md) · [Training guide](docs/TRAINING.md) · [Paper figure map](docs/PAPER_RESULTS.md) · [Data dictionary](data/README.md)

Research software and numerical data accompanying **A generative framework for turbomachinery aerodynamic inverse design based on flow matching with classifier-free guidance** by Yueteng Wu, Zihao Zhu, Juan Du, Xinglong Zhang, Penghao Duan and Yangyang Wang.

The framework maps three aerodynamic targets to 21 rotor geometry parameters. A Transformer encoder surrogate (TeM) supports performance prediction and data augmentation. Generated candidates can subsequently undergo independent CFD assessment.

## Included models

| Identifier | Model | Role |
|---|---|---|
| `fm_cfg` | cFM-CFG | Proposed conditional flow-matching method |
| `ar` | cAR | Autoregressive Transformer baseline |
| `cvae` | cVAE | Conditional variational autoencoder baseline |
| `cgan` | cWGAN-GP | Conditional Wasserstein GAN baseline |
| `ddpm` | cDDPM | Conditional diffusion baseline |
| `cinn` | cINN | Conditional invertible-network baseline |
| `performance_predictor` | TeM | Transformer performance predictor |
| `fm` | Standard FM | No-CFG ablation in Fig. 9d–f |

Only models used in the paper are part of the current release. `--model all` trains the six comparison generators sequentially; TeM and standard FM have separate commands.

## Install

Reference environment: **Python 3.9.18, PyTorch 2.3.0**, with CUDA 11.8 for the original GPU environment. Use a clean environment and install a suitable PyTorch build for your hardware. The commands below are the same on Windows and Linux:

```bash
git clone https://github.com/wuyueteng/Turbomachinery-Inverse-Design-FM-CFG.git
cd Turbomachinery-Inverse-Design-FM-CFG
python -m pip install -r requirements.txt
python scripts/check_data.py --check-results
python scripts/smoke_check.py
```

CUDA is recommended for full training. Use `--device cpu` for small software checks, or `--device cuda:0` to select a GPU. The default `auto` selects CUDA when available. All commands below run from the repository root; no source-code edits are needed.

## Train, generate and evaluate

Datasets are included. Pretrained weights are **not distributed**; users train their own networks. The following commands use the supplied selected configurations without running hyperparameter search first:

```bash
python scripts/train.py --model performance_predictor --output-dir runs/tem
python scripts/train.py --model fm_cfg --output-dir runs/fm_cfg
python scripts/inference.py --run-dir runs/fm_cfg --predictor-run runs/tem
```

The last command generates candidates for 100 held-out targets, with 10 designs per target, and evaluates their performance using the trained surrogate. It writes geometry arrays, target/predicted performance and metrics under `runs/fm_cfg/inference/`. Omit `--predictor-run` for generation only. These are surrogate predictions, not CFD solutions.

Train the six-model comparison or the no-CFG ablation:

```bash
python scripts/train.py --model all --output-dir runs/comparison
python scripts/train.py --model fm --output-dir runs/fm
```

Every run saves its actual configuration, split indices, input hashes, loss history, `best.pt` for inference and `last.pt` for continuation. The training defaults retain the selected architecture and original optimizer/validation rules. Changing a configuration creates a new experiment; it does not modify archived paper results.

```bash
python scripts/train.py --model fm_cfg --epochs 300 --batch-size 128 --device cuda:0 --output-dir runs/custom
python scripts/train.py --resume runs/custom --device cuda:0
```

See [the training guide](docs/TRAINING.md) for custom JSON configurations, exact continuation, training seeds, raw target input, HPO and software tests. Reduced batches/epochs are useful for checking installation; they are not paper-performance runs.

## Reproduce paper plots without training

```bash
python scripts/verify_paper_results.py
python scripts/reproduce_figures.py --all
```

Archived numerical outputs are under `paper_results/`. The command regenerates network-related figures as PNG/PDF under `outputs/paper_figures/`, including mixed-figure network panels only. Use `--figure 2 3 6` to select figures. The [figure map](docs/PAPER_RESULTS.md) identifies every supported figure and its sources. The replotting entrypoint preserves numerical definitions while using portable layout/fonts; it does not recreate CFD plots or require network weights.

## Data and provenance

- Dataset I: **3,840 CFD-labelled geometry/performance pairs**.
- Additional set: **10,000 geometries with TeM-predicted labels**.
- Dataset II: **13,840 pairs**, in the original stored order.

The historical `_5200` filenames are retained for code compatibility. Labels are ordered **mass flow (kg/s), isentropic efficiency, total pressure ratio**. Input arrays, lossless CSV exports and a row-level source mapping are provided; see [data/README.md](data/README.md). Predictor and generator normalization uses the CFD reference scale, as implemented in the supplied research code.

The repository includes the benchmark, uncertainty, robustness, HPO histories, trajectories, ablations and cost measurements corresponding to the network analyses. New training outputs are stored separately. Full retraining is stochastic and may differ from the archived realization; plots reconstructed from archived data retain the reported values. The robustness archive varies training seeds on a fixed data split.

## Layout

```text
configs/        Selected model and training settings
models/         Paper network definitions and original training entrypoints
shared/         Loading, evaluation and run-based training
scripts/        Training, generation, HPO, verification and plotting
experiments/    Original network-analysis workflows
data/           Original and augmented datasets, schema and provenance
paper_results/  Archived numerical results, source hashes and figure previews
tests/          Short training, resume and CLI integration tests
docs/           Training, implementation, validation and figure mapping
```

## Citation and license

Use [CITATION.cff](CITATION.cff) to cite this software release and identify the associated manuscript. The manuscript title is provided without an invented publication DOI. Code is licensed under [MIT](LICENSE); datasets, numerical results and result figures are licensed under [CC BY 4.0](LICENSE-DATA). Third-party notices remain in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Browse the [reconstructed PNG/PDF figure gallery](paper_results/previews/README.md).
