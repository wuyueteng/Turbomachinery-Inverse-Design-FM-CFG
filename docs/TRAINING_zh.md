# 训练与推理指南

## 参数设置

`configs/<模型名>.json` 包含网络参数、训练轮数、批量、早停、训练种子、划分种子及数据集选择。`--config` 指定自定义 JSON，网络参数按键合并；显式命令行参数优先。例如 `{"model_params":{"hidden_dim":256,"n_blocks":4}}` 可调整 cFM-CFG 规模，属于新的实验设置。

运行 `python scripts/train.py --help` 查看全部参数。支持模型、配置、轮数、批量、学习率、设备、CPU 线程、训练与划分种子、数据路径、输出路径、早停及续训。cWGAN-GP 的 `--lr` 同时设置两个优化器学习率，也可在配置中分别设置 `lr_g`、`lr_d`。

生成模型默认使用增强数据、300 轮、批量 256、训练种子及 NumPy 划分种子 42，以及已选参数；非 GAN 模型按原规则验证和早停，GAN 保留交替更新且不早停。预测器使用 CFD 数据、300 轮、批量 32、学习率 0.0001、训练种子及 Torch 划分种子 15，不早停。标准 FM 使用原消融设置。

训练生成模型本身不需要代理权重；代理评估与下游误差寻优需要先训练 TeM。预测器训练沿用 80/10/10 划分及第二子集作为验证集的实现；最后子集保留。稳健性实验保留其独立脚本中的固定划分和验证子集定义。

## 输出与续训

未指定输出目录时自动建立带时间戳的 `runs/<模型名>/` 子目录；训练六模型时分别保存。新运行拒绝写入已有非空目录。

每次保存 `config.json`（实际参数）、`data_manifest.json`（输入校验值）、`split_indices.npz`（划分索引）、`history.json`（损失历史）、`best.pt`（最佳验证权重与归一化）和 `last.pt`（完整续训状态）。

```bash
python scripts/train.py --model fm_cfg --epochs 300 --stop-after 10 --output-dir runs/continued
python scripts/train.py --resume runs/continued
```

第一条命令只运行本次的 10 轮，仍按 300 轮计划调度学习率。续训恢复网络、优化器、调度器、早停计数及 Python、NumPy、Torch 随机状态；GAN 同时恢复两个优化器与更新计数。输入数据通过校验值核对。

续训沿用原配置，允许调整设备、数据位置及本次停止轮数；其他参数修改需创建新运行。同一 CPU 环境已测试连续训练与续训一致，跨设备结果不保证逐位相同。只加载可信的续训文件。

## 自定义目标与评估

```bash
python scripts/inference.py --run-dir runs/fm_cfg --predictor-run runs/tem --n-cond 100 --n-samples 10
python scripts/inference.py --run-dir runs/fm_cfg --targets targets.csv --n-samples 20 --output-dir outputs/my_candidates
```

目标文件为无表头 CSV 或 NumPy 数组，形状 `(N,3)`，物理量顺序为质量流量、效率、压比；单目标为 `(1,3)`。未提供目标文件时，从运行保存的测试索引中选取前 `--n-cond` 个目标。提供目标文件时使用其中所有行。

输出归一化与物理几何参数、重复后的目标、组合 CSV 和来源说明；指定预测器后还输出预测性能及各变量指标。目标方差或极差为零时，R² 或 nRMSE 记录为 `null`。CFD 验证属于独立流程，代理输出不会标记为 CFD 值。

## 可选超参数寻优

```bash
python scripts/optuna_hpo.py --model fm_cfg --n_trials 50 --retrain --predictor-run runs/tem
python models/performance_predictor/optuna_hpo.py --n-trials 100 --epochs 300
```

生成模型寻优沿用原下游代理误差目标，`--n_trials` 是原脚本定义的已完成试验数目标，剪枝可能增加实际尝试数。默认目标 40，命令中显式设置为 50。归档寻优记录保留其真实试验数。原寻优入口将输出写入相应模型目录，论文结果另存于 `paper_results/`。

预测器架构寻优单独提供，输出的架构与权重成对使用；不同架构的结果不标作论文固定架构。已有旧式权重可通过 `scripts/inference.py --model fm_cfg` 使用；原实验脚本可通过环境变量 `TURBO_PREDICTOR_CHECKPOINT` 指定预测器 `best.pt`。

## 常见问题

- 显卡不可用：检查 PyTorch 安装，或指定 `--device cpu`。
- 显存不足：减小 `--batch-size`；缩小网络需通过配置进行，并作为新实验记录。
- 缺少数据：运行 `scripts/check_data.py`，核对 `--data-dir` 与文件名。
- 输出目录非空：使用新目录；如继续原训练则使用 `--resume`。
- 缺少预测器：先训练 `performance_predictor`，再传入其运行目录。
- 快速检查可使用 `--epochs 1 --max-batches 1`，减少的预算会记录在配置中，不作为论文精度结果。
- 运行 `python scripts/smoke_check.py` 和 `python -m unittest discover -s tests -v` 检查网络接口、短训练、续训及命令行流程。测试不重复全部科研训练。
