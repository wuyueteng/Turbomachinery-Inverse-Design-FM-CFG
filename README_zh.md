# 基于流匹配与无分类器引导的涡轮机械逆设计

[English](README.md) · [训练指南](docs/TRAINING_zh.md) · [论文图表对应表](docs/PAPER_RESULTS.md) · [数据说明](data/README.md)

本仓库提供论文 *A generative framework for turbomachinery aerodynamic inverse design based on flow matching with classifier-free guidance* 的网络代码、原始及增强数据，以及神经网络分析相关结果。

## 包含的模型

| 命令名称 | 模型与用途 |
|---|---|
| `fm_cfg` | 本文 cFM-CFG 方法 |
| `ar`、`cvae`、`cgan`、`ddpm`、`cinn` | 五个基线：cAR、cVAE、cWGAN-GP、cDDPM、cINN |
| `performance_predictor` | Transformer 性能预测器 TeM |
| `fm` | 图 9d–f 使用的不带 CFG 的标准 FM 消融模型 |

`--model all` 只训练六个主要生成模型；预测器和标准 FM 分别训练。当前发布内容仅包含论文实际使用的模型。

## 安装与数据检查

参考环境为 Python 3.9.18、PyTorch 2.3.0，原 GPU 环境使用 CUDA 11.8。建议创建独立环境，安装适合显卡的 PyTorch，再执行以下命令。所有命令均在仓库根目录运行，Windows 和 Linux 使用相同的 Python 命令。

```bash
git clone https://github.com/wuyueteng/Turbomachinery-Inverse-Design-FM-CFG.git
cd Turbomachinery-Inverse-Design-FM-CFG
python -m pip install -r requirements.txt
python scripts/check_data.py --check-results
python scripts/smoke_check.py
```

仓库已包含数据，不需要手动调整源代码中的路径。完整训练建议使用 GPU；`--device cpu` 可用于短流程检查，`--device cuda:0` 指定 GPU，默认自动选择可用设备。

## 从零训练、生成和评估

本仓库不分发预训练权重。默认直接使用已选定的参数，用户无需先做超参数搜索：

```bash
python scripts/train.py --model performance_predictor --output-dir runs/tem
python scripts/train.py --model fm_cfg --output-dir runs/fm_cfg
python scripts/inference.py --run-dir runs/fm_cfg --predictor-run runs/tem
```

第三步默认从保留测试划分中取 100 个性能目标，每个目标生成 10 个设计，并使用已训练的 TeM 评估，输出保存于 `runs/fm_cfg/inference/`。省略 `--predictor-run` 可只生成几何。代理预测与 CFD 求解结果在数据说明中分别标识。

训练全部主要对比模型或单独训练消融模型：

```bash
python scripts/train.py --model all --output-dir runs/comparison
python scripts/train.py --model fm --output-dir runs/fm
```

每次运行保存实际参数、数据划分、输入校验值、损失历史、推理所用 `best.pt` 和续训所用 `last.pt`。不同实验使用独立目录，不会覆盖论文归档结果。

```bash
python scripts/train.py --model fm_cfg --epochs 300 --batch-size 128 --device cuda:0 --output-dir runs/custom
python scripts/train.py --resume runs/custom --device cuda:0
```

完整参数、配置文件、续训、目标输入、寻优及故障处理见[中文训练指南](docs/TRAINING_zh.md)。缩短训练轮数或减少批次只用于检查运行流程，不代表论文精度。

## 不训练网络，直接重绘论文结果

```bash
python scripts/verify_paper_results.py
python scripts/reproduce_figures.py --all
```

结果保存在 `outputs/paper_figures/`，同时输出 PNG 和 PDF。可用 `--figure 2 3 6` 选择图号。重绘使用归档数值，不需要权重；采用可移植的字体和布局，混合图仅重绘网络相关面板。对应关系见[图表清单](docs/PAPER_RESULTS.md)。

## 数据与复现说明

发布数据包括 3,840 组 CFD 原始几何—性能配对、10,000 组新增几何及 TeM 预测标签，以及保持原顺序的 13,840 组合并数据。提供 NumPy、无损 CSV、字段说明和样本来源映射。

性能列顺序始终为：**质量流量、等熵效率、总压比**。`_5200` 是历史文件名，实际 CFD 样本数为 3,840。归一化保留原实现的完整 CFD 参考尺度；稳健性实验在固定划分下改变训练随机种子。

网络结构、损失及采样方法保留研究实现。训练有随机性，新训练结果与论文归档结果分开保存；归档数据用于核对和重绘报告结果。

代码采用 [MIT](LICENSE)，数据、结果及结果图采用 [CC BY 4.0](LICENSE-DATA)。引用信息见 [CITATION.cff](CITATION.cff)，第三方署名见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

可直接查看[重绘图件预览（PNG／PDF）](paper_results/previews/README.md)。
