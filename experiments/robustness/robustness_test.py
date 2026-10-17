"""
稳健性验证脚本
通过100个不同的随机数种子训练模型，验证模型的稳健性
针对每个变量分别计算各项指标（均值和标准差）
支持多进程并行训练

输出变量:
- Mass: 质量流量
- Eff: 效率
- Pre: 压比

输出指标:
- MAE / MAE_std: 绝对误差的均值和标准差
- MSE / MSE_std: 平方误差的均值和标准差
- RMSE: 均方根误差 = sqrt(MSE)
- nRMSE: 归一化均方根误差 = RMSE / (max(label) - min(label))
- MRE / MRE_std: 相对误差的均值和标准差
- R2: 决定系数 (sklearn.metrics.r2_score)
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from models.performance_predictor.vit import ViT
from models.performance_predictor.dataset import PerformanceDataset as input_dataset
from sklearn.metrics import r2_score


def set_seed(seed):
    """设置所有随机数种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data(data_file, label_file, batch_size=32, seed=15):
    """加载数据集"""
    dataset = input_dataset(data_file, label_file)
    train_size = int(0.8 * len(dataset))
    test_size = int(0.1 * len(dataset))
    validation_size = len(dataset) - train_size - test_size

    # 使用固定种子划分数据集，保证每次划分一致
    generator = torch.Generator().manual_seed(seed)
    train_dataset, test_dataset, validation_dataset = random_split(
        dataset, [train_size, test_size, validation_size], generator=generator
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

    # 返回数据集的归一化参数用于反归一化
    return train_loader, validation_loader, dataset.max1, dataset.min1


def build_model(device):
    """构建模型 - 使用与原始训练脚本相同的配置"""
    model = ViT(
        lenth=1,
        channels=21,
        num_classes=3,
        dim=256,
        depth=14,
        heads=12,
        mlp_dim=256,
        dropout=0.085,
        emb_dropout=0.072,
        dim_head=256
    ).to(device)
    return model


def calculate_metrics(preds, labels, label_max, label_min):
    """
    计算评估指标（针对每个变量分别计算）
    preds, labels: 归一化后的预测值和真实值
    label_max, label_min: 用于反归一化
    返回: 每个变量的指标（均值和标准差）

    指标说明:
    - MAE: 平均绝对误差 = mean(|residuals|)
    - MAE_std: 绝对误差的标准差 = std(|residuals|)
    - MSE: 均方误差 = mean(residuals²)
    - MSE_std: 平方误差的标准差 = std(residuals²)
    - RMSE: 均方根误差 = sqrt(MSE)
    - nRMSE: 归一化均方根误差 = RMSE / (max(label) - min(label))
    - MRE: 平均相对误差 = mean(|residuals| / |labels|)
    - MRE_std: 相对误差的标准差 = std(|residuals| / |labels|)
    - R2: 决定系数

    变量说明:
    - Mass: 质量流量
    - Eff: 效率
    - Pre: 压比
    """
    # 反归一化
    preds_denorm = preds * (label_max - label_min) + label_min
    labels_denorm = labels * (label_max - label_min) + label_min

    # 变量名称
    var_names = ['Mass', 'Eff', 'Pre']

    metrics = {}

    # 针对每个变量分别计算指标
    for i, var_name in enumerate(var_names):
        pred_i = preds_denorm[:, i]
        label_i = labels_denorm[:, i]

        # 残差
        residuals = pred_i - label_i

        # 绝对误差
        abs_errors = np.abs(residuals)
        mae = np.mean(abs_errors)
        mae_std = np.std(abs_errors)

        # 平方误差
        sq_errors = residuals ** 2
        mse = np.mean(sq_errors)
        mse_std = np.std(sq_errors)

        # RMSE
        rmse = np.sqrt(mse)

        # nRMSE: 归一化均方根误差 = RMSE / (max - min)
        label_range = np.max(label_i) - np.min(label_i)
        nrmse = rmse / (label_range + 1e-8)

        # MRE: 平均相对误差 = mean(|residuals| / |labels|)
        relative_errors = abs_errors / (np.abs(label_i) + 1e-8)
        mre = np.mean(relative_errors)
        mre_std = np.std(relative_errors)

        # R2
        r2 = r2_score(label_i, pred_i)

        metrics[var_name] = {
            'MAE': mae,
            'MAE_std': mae_std,
            'MSE': mse,
            'MSE_std': mse_std,
            'RMSE': rmse,
            'nRMSE': nrmse,
            'MRE': mre,
            'MRE_std': mre_std,
            'R2': r2
        }

    return metrics


def train_single_seed(seed, data_file, label_file, n_epochs=300, device_id=0):
    """
    使用单个随机种子训练模型并返回验证集指标
    """
    device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"

    # 设置随机种子
    set_seed(seed)

    # 加载数据 - 使用固定的数据划分种子15，只改变模型初始化和训练的随机性
    train_loader, val_loader, label_max, label_min = load_data(
        data_file, label_file, batch_size=32, seed=15
    )

    # 构建模型
    model = build_model(device)

    # 训练配置
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-6)

    best_val_loss = float('inf')
    best_preds = None
    best_labels = None

    # 训练循环
    for epoch in range(n_epochs):
        # 训练
        model.train()
        for batch in train_loader:
            x, y = batch['data'].to(device), batch['label'].to(device)
            pred = model(x)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

        # 验证
        model.eval()
        val_preds = []
        val_labels = []
        val_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                x, y = batch['data'].to(device), batch['label'].to(device)
                pred = model(x)
                loss = loss_fn(pred, y)
                val_loss += loss.item()
                val_preds.append(pred.cpu().numpy())
                val_labels.append(y.cpu().numpy())

        val_loss /= len(val_loader)

        # 保存最佳模型的预测结果
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_preds = np.concatenate(val_preds, axis=0)
            best_labels = np.concatenate(val_labels, axis=0)

    # 计算最终指标
    metrics = calculate_metrics(best_preds, best_labels, label_max, label_min)
    metrics['seed'] = seed

    return metrics


def train_worker(args):
    """多进程worker函数"""
    seed, data_file, label_file, n_epochs, device_id = args
    try:
        result = train_single_seed(seed, data_file, label_file, n_epochs, device_id)
        return result
    except Exception as e:
        return {'seed': seed, 'error': str(e)}


def run_robustness_test(num_seeds=100, n_epochs=300, num_workers=4, start_seed=0):
    """
    运行稳健性测试
    num_seeds: 随机种子数量
    n_epochs: 每个种子训练的轮数
    num_workers: 并行训练的进程数
    start_seed: 起始种子值
    """
    data_file = str(ROOT / "data/parameter_geometry_5200.npy")
    label_file = str(ROOT / "data/label_5200.npy")

    # 检查GPU数量
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("警告: 未检测到GPU，将使用CPU训练")
        num_gpus = 1

    print(f"检测到 {num_gpus} 个GPU")
    print(f"将使用 {num_workers} 个并行进程")
    print(f"总共训练 {num_seeds} 个不同的随机种子")
    print(f"每个种子训练 {n_epochs} 轮")
    print("-" * 50)

    # 准备任务参数
    seeds = list(range(start_seed, start_seed + num_seeds))
    tasks = [(seed, data_file, label_file, n_epochs, seed % num_gpus) for seed in seeds]

    results = []

    # 使用进程池并行训练
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(train_worker, task): task[0] for task in tasks}

        with tqdm(total=num_seeds, desc="训练进度") as pbar:
            for future in as_completed(futures):
                seed = futures[future]
                try:
                    result = future.result()
                    results.append(result)

                    if 'error' not in result:
                        pbar.set_postfix({
                            'seed': seed,
                            'Eff_R2': f"{result['Eff']['R2']:.4f}",
                            'Pre_R2': f"{result['Pre']['R2']:.4f}",
                            'Eff_MRE': f"{result['Eff']['MRE']:.4f}"
                        })
                    else:
                        pbar.set_postfix({'seed': seed, 'status': 'error'})

                except Exception as e:
                    results.append({'seed': seed, 'error': str(e)})

                pbar.update(1)

    return results


def analyze_and_save_results(results, output_dir):
    """分析结果并保存到CSV文件"""
    # 过滤掉错误的结果
    valid_results = [r for r in results if 'error' not in r]

    if len(valid_results) == 0:
        print("没有有效的训练结果!")
        return

    print(f"\n成功训练 {len(valid_results)}/{len(results)} 个种子")

    var_names = ['Mass', 'Eff', 'Pre']
    metric_names = ['MAE', 'MAE_std', 'MSE', 'MSE_std', 'RMSE', 'nRMSE', 'MRE', 'MRE_std', 'R2']

    # 保存CSV文件（每行一个case）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = os.path.join(output_dir, f'robustness_results_{timestamp}.csv')

    with open(csv_file, 'w', encoding='utf-8') as f:
        # 写入表头
        headers = ['seed']
        for var in var_names:
            for metric in metric_names:
                headers.append(f'{var}_{metric}')
        f.write(','.join(headers) + '\n')

        # 写入每次实验的数据（每行一个case）
        for r in valid_results:
            row = [str(r['seed'])]
            for var in var_names:
                for metric in metric_names:
                    if metric == 'R2':
                        row.append(f"{r[var][metric]:.6f}")
                    else:
                        row.append(f"{r[var][metric]:.8f}")
            f.write(','.join(row) + '\n')

        # 写入空行分隔
        f.write('\n')

        # 写入跨种子的统计（所有种子的均值和标准差）
        f.write(','.join(headers) + '\n')

        # Mean行（跨种子均值）
        mean_row = ['Mean']
        for var in var_names:
            for metric in metric_names:
                vals = [r[var][metric] for r in valid_results]
                if metric == 'R2':
                    mean_row.append(f"{np.mean(vals):.6f}")
                else:
                    mean_row.append(f"{np.mean(vals):.8f}")
        f.write(','.join(mean_row) + '\n')

        # Std行（跨种子标准差）
        std_row = ['Std']
        for var in var_names:
            for metric in metric_names:
                vals = [r[var][metric] for r in valid_results]
                if metric == 'R2':
                    std_row.append(f"{np.std(vals):.6f}")
                else:
                    std_row.append(f"{np.std(vals):.8f}")
        f.write(','.join(std_row) + '\n')

    # 打印结果摘要
    print("\n" + "=" * 80)
    print("稳健性测试结果摘要")
    print("=" * 80)

    for var in var_names:
        print(f"\n【{var}】")
        for metric in metric_names:
            vals = [r[var][metric] for r in valid_results]
            if metric == 'R2':
                print(f"  {metric:>8s}:  Mean={np.mean(vals):.4f}  ±  Std={np.std(vals):.4f}")
            else:
                print(f"  {metric:>8s}:  Mean={np.mean(vals):.6f}  ±  Std={np.std(vals):.6f}")

    print(f"\n结果已保存到: {csv_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='模型稳健性测试')
    parser.add_argument('--num_seeds', type=int, default=100, help='随机种子数量 (默认: 100)')
    parser.add_argument('--n_epochs', type=int, default=300, help='每个种子训练轮数 (默认: 300)')
    parser.add_argument('--num_workers', type=int, default=4, help='并行进程数 (默认: 2)')
    parser.add_argument('--start_seed', type=int, default=0, help='起始种子值 (默认: 0)')

    args = parser.parse_args()

    output_dir = os.path.dirname(os.path.abspath(__file__))

    print("=" * 60)
    print("模型稳健性测试")
    print("=" * 60)
    print(f"随机种子数量: {args.num_seeds}")
    print(f"每个种子训练轮数: {args.n_epochs}")
    print(f"并行进程数: {args.num_workers}")
    print(f"起始种子: {args.start_seed}")
    print(f"结果保存目录: {output_dir}")
    print("=" * 60)

    # 运行测试
    results = run_robustness_test(
        num_seeds=args.num_seeds,
        n_epochs=args.n_epochs,
        num_workers=args.num_workers,
        start_seed=args.start_seed
    )

    # 分析并保存结果
    analyze_and_save_results(results, output_dir)

    print("\n测试完成!")
