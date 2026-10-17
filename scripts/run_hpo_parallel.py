"""
双 GPU 并行超参数寻优
======================
将 7 个模型分配到 2 个 GPU 并行运行。
同一 GPU 上的模型串行执行 (避免显存争抢)。

Usage:
    python scripts/run_hpo_parallel.py
    python scripts/run_hpo_parallel.py --n_trials 50
    python scripts/run_hpo_parallel.py --retrain
"""

import argparse
import os
import sys
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HPO_SCRIPT = os.path.join(ROOT_DIR, "scripts", "optuna_hpo.py")
PYTHON = sys.executable

# GPU 分组: 按计算量均衡分配
# GPU 分组: 按计算量均衡分配 (DDPM 类较慢, 分到不同 GPU)
GPU_GROUPS = {
    0: ["fm_cfg", "cvae", "cinn"],
    1: ["ddpm", "cgan", "ar"],
}


def run_gpu_group(gpu_id, models, n_trials, retrain):
    """在指定 GPU 上串行运行一组模型"""
    results = []
    for model in models:
        print(f"\n[GPU {gpu_id}] === {model} (n_trials={n_trials}) ===")
        cmd = [
            PYTHON, HPO_SCRIPT,
            "--model", model,
            "--n_trials", str(n_trials),
            "--gpu", str(gpu_id),
        ]
        if retrain:
            cmd.append("--retrain")

        t0 = time.time()
        ret = subprocess.run(cmd, cwd=ROOT_DIR)
        elapsed = time.time() - t0
        status = "OK" if ret.returncode == 0 else "FAIL"
        results.append((model, status, elapsed))
        print(f"[GPU {gpu_id}] {model}: {status} ({elapsed/60:.1f} min)")

    return results


def main():
    parser = argparse.ArgumentParser(description="双 GPU 并行 HPO")
    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument("--retrain", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("  Dual-GPU Parallel HPO")
    print("=" * 60)
    for gpu_id, models in GPU_GROUPS.items():
        print(f"  GPU {gpu_id}: {models}")
    print(f"  n_trials: {args.n_trials}")
    print(f"  retrain:  {args.retrain}")
    print("=" * 60)

    t_start = time.time()

    # 2 个 GPU 并行, 每个 GPU 内串行
    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = {}
        for gpu_id, models in GPU_GROUPS.items():
            f = executor.submit(run_gpu_group, gpu_id, models, args.n_trials, args.retrain)
            futures[f] = gpu_id

        all_results = []
        for f in as_completed(futures):
            gpu_id = futures[f]
            try:
                results = f.result()
                all_results.extend(results)
                print(f"\n[GPU {gpu_id}] 全部完成")
            except Exception as e:
                print(f"\n[GPU {gpu_id}] 异常: {e}")

    total_time = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  总耗时: {total_time/3600:.1f} h")
    print(f"{'='*60}")
    for model, status, elapsed in sorted(all_results, key=lambda x: x[0]):
        print(f"  {model:<16s} {status:>6s}  {elapsed/60:6.1f} min")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
