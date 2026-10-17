# 涡轮机械气动逆设计研究代码

本仓库对应论文 *A generative framework for turbomachinery aerodynamic inverse design based on flow matching with classifier-free guidance*。

包含六个主要对比模型（cFM-CFG、cAR、cVAE、cWGAN-GP、cDDPM、cINN）、三个消融或扩展模型、Transformer 性能预测、Optuna 超参数寻优、生成与评估，以及消融、生成轨迹、计算耗时、稳健性和 MC dropout 分析代码。

## 使用顺序

1. 安装与硬件匹配的 PyTorch，再执行 `python -m pip install -r requirements.txt`。
2. 执行 `python scripts/smoke_check.py`，检查网络接口。
3. 按 [数据说明](data/README.md) 放置数据；性能列顺序为质量流量、等熵效率、总压比。
4. 运行 `python models/performance_predictor/train.py --epochs 300`，或放置已有预测器权重。
5. 运行 `python scripts/optuna_hpo.py --model fm_cfg --n_trials 50 --retrain`。
6. 运行 `python scripts/inference.py --model fm_cfg --n_cond 100 --n_samples 10`。

模型参数、损失和采样算法保留原项目实现。发布整理主要处理路径、训练输出目录、依赖说明及模块导航。现有 HPO JSON 保留原记录，命令中通过 `--n_trials` 明确指定新实验的预算。

原始数据、预训练权重和投稿材料未上传；数据沿用向通讯作者合理申请的方式提供。详细运行说明见 [English README](README.md)，实现说明见 [implementation notes](docs/implementation.md)。
