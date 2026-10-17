"""Profile saved model configurations without changing them.

Each model/phase runs in its own process, sequentially on one GPU. Training
updates only a temporary in-memory copy. Inference uses the untouched checkpoint.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
MODELS = ["fm_cfg", "ar", "cvae", "cgan", "ddpm", "cinn"]
DISPLAY = dict(zip(MODELS, ["cFM-CFG", "cAR", "cVAE", "cWGAN-GP", "cDDPM", "cINN"]))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def gpu_status():
    return subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,name,uuid,driver_version,memory.total,memory.used,utilization.gpu,temperature.gpu",
        "--format=csv"], text=True).strip()


def model_paths(name):
    base = ROOT / "models" / name
    return (base / "results" / f"hpo_best_{name}.json",
            base / "checkpoints" / f"{name}_hpo.pth")


def configure(seed, gpu):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(gpu)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    return torch.device(f"cuda:{gpu}")


def worker(args):
    import numpy as np
    import torch
    from shared.data_utils import get_dataloaders
    import scripts.optuna_hpo as hpo

    out = Path(args.output)
    result_path = out / "raw" / f"{args.model}_{args.phase}.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    device = configure(42, args.gpu)
    hpo.DEVICE = device
    cfg_path, ckpt_path = model_paths(args.model)
    cfg = read_json(cfg_path)
    model = hpo.create_model(args.model, cfg["best_params"], device=device)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    del state
    model.to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable == cfg["best_n_params"], (trainable, cfg["best_n_params"])
    if args.model == "cgan":
        active = sum(p.numel() for p in model.generator.parameters())
    elif args.model == "cvae":
        active = sum(p.numel() for p in model.decoder.parameters())
    else:
        active = trainable
    result = {
        "model": args.model, "display_name": DISPLAY[args.model], "phase": args.phase,
        "started_local": datetime.now().isoformat(),
        "environment": {"platform": platform.platform(), "python": platform.python_version(),
                        "pytorch": torch.__version__, "cuda": torch.version.cuda,
                        "cudnn": torch.backends.cudnn.version(), "gpu_index": args.gpu,
                        "gpu_name": torch.cuda.get_device_name(device), "cpu_threads": 4,
                        "precision": "float32", "matmul_tf32": False, "cudnn_tf32": False,
                        "gpu_before": gpu_status()},
        "checkpoint_sha256": sha256(ckpt_path), "config_sha256": sha256(cfg_path),
        "trainable_parameters": trainable, "inference_active_parameters": active,
        "buffer_elements": sum(b.numel() for b in model.buffers()), "best_params": cfg["best_params"],
        "memory_definition": "Peak PyTorch allocated tensor memory; full model resident. Context/driver and allocator cache excluded.",
    }
    rows = []
    if args.phase == "train":
        loader, _, _, info = get_dataloaders(batch_size=256, use_merged=True)
        optimizer = None if args.model == "cgan" else torch.optim.Adam(
            model.parameters(), lr=cfg["best_params"]["lr"])
        scheduler = None if optimizer is None else torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg["hpo_config"]["max_epochs"], eta_min=1e-6)
        result["protocol"] = {
            "train_split_size": info["n_train"], "val_split_size": info["n_val"],
            "test_split_size": info["n_test"], "batch_size": 256,
            "batches_per_epoch": len(loader), "processed_samples_per_epoch": len(loader) * 256,
            "drop_last": True, "num_workers": 0, "pin_memory": False,
            "warmup_epochs": 2, "measured_epochs": 10,
            "epoch_scope": "CPU DataLoader iteration, host-to-device transfer, original loss/backward/optimizer, loss logging, scheduler; excludes validation, evaluation, checkpoint I/O, HPO.",
            "initialization": "Saved checkpoint weights; fresh Adam state; updates discarded.",
            "training_function": "scripts.optuna_hpo.train_one_epoch (unchanged)",
        }
        for i in range(12):
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            step_before = getattr(model, "_step_count", None)
            start = time.perf_counter()
            loss = hpo.train_one_epoch(model, loader, optimizer, args.model, epoch=i)
            if scheduler is not None:
                scheduler.step()
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start
            row = {"epoch": i + 1, "warmup": i < 2, "seconds": elapsed, "loss": float(loss),
                   "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                   "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}
            if args.model == "cgan":
                interval = range(step_before, model._step_count)
                row["generator_updates"] = sum(k % (model.n_critic + 1) == model.n_critic for k in interval)
                row["discriminator_updates"] = len(loader) - row["generator_updates"]
            assert np.isfinite(loss), row
            rows.append(row)
            print(f"{args.model} train epoch {i+1:02d}: {elapsed:.3f}s, loss={loss:.6f}", flush=True)
        result["trained_copy_saved"] = False
    else:
        conditions = np.load(out / "inputs" / "conditions_norm.npy")
        model.eval()
        if hasattr(model, "temperature"):
            model.temperature = 0.8
        result["protocol"] = {
            "conditions": 100, "designs_per_condition": 10, "total_designs": 1000,
            "calls_per_repetition": 100, "warmup_calls": 5, "measured_repetitions": 3,
            "sampling_seed_per_repetition": [42, 43, 44],
            "temperature_ar_cinn": 0.8,
            "inference_scope": "Original public sample(), 100 sequential calls of 10 designs; includes noise, input transfer, clamping and CPU numpy return. Excludes model/data loading, TeM, CFD, file I/O.",
            "full_model_on_gpu": True, "sampling_implementation_modified": False,
        }
        for _ in range(5):
            warm = model.sample(conditions[0], n_samples=10)
            assert warm.shape == (10, 21) and np.isfinite(warm).all()
        for rep in range(3):
            torch.manual_seed(42 + rep)
            torch.cuda.manual_seed_all(42 + rep)
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
            generated = [model.sample(c, n_samples=10) for c in conditions]
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start
            peak = torch.cuda.max_memory_allocated(device)
            reserved = torch.cuda.max_memory_reserved(device)
            generated = np.vstack(generated)
            assert generated.shape == (1000, 21)
            assert np.isfinite(generated).all()
            assert generated.min() >= 0 and generated.max() <= 1
            row = {"repetition": rep + 1, "warmup": False, "seconds": elapsed,
                   "ms_per_design_amortized": elapsed,  # seconds / 1000 * 1000
                   "peak_allocated_bytes": peak, "peak_reserved_bytes": reserved,
                   "output_shape": list(generated.shape), "all_finite": True,
                   "output_min": float(generated.min()), "output_max": float(generated.max())}
            rows.append(row)
            print(f"{args.model} inference {rep+1}/3: {elapsed:.3f}s / 1000 designs", flush=True)
    result["measurements"] = rows
    measured = [r for r in rows if not r["warmup"]]
    result["summary"] = {
        "seconds_mean": float(np.mean([r["seconds"] for r in measured])),
        "seconds_sd": float(np.std([r["seconds"] for r in measured], ddof=1)),
        "peak_allocated_mib": max(r["peak_allocated_bytes"] for r in measured) / 2**20,
        "peak_reserved_mib": max(r["peak_reserved_bytes"] for r in measured) / 2**20,
    }
    result["checkpoint_unchanged"] = sha256(ckpt_path) == result["checkpoint_sha256"]
    assert result["checkpoint_unchanged"]
    result["finished_local"] = datetime.now().isoformat()
    result["environment"]["gpu_after"] = gpu_status()
    write_json(result_path, result)


def main(args):
    import numpy as np
    from shared.data_utils import get_dataloaders
    out = Path(args.output).resolve()
    if out.exists():
        raise FileExistsError(f"Use a new output directory: {out}")
    (out / "inputs").mkdir(parents=True)
    (out / "raw").mkdir()
    (out / "logs").mkdir()
    _, _, _, info = get_dataloaders(batch_size=256, use_merged=True)
    eval_idx = np.random.RandomState(42).choice(len(info["test_label"]), 100, replace=False)
    conditions = info["test_label"][eval_idx]
    np.save(out / "inputs" / "conditions_norm.npy", conditions)
    np.save(out / "inputs" / "test_condition_indices.npy", eval_idx)
    stats = info["stats"]
    target = np.repeat(conditions, 10, axis=0) * (stats["label_max"] - stats["label_min"]) + stats["label_min"]
    target_errors = {}
    sources = [ROOT / "data" / name for name in
               ("merged_geometry.npy", "merged_label.npy", "parameter_geometry_5200.npy", "label_5200.npy")]
    sources += [ROOT / "scripts" / "optuna_hpo.py", ROOT / "scripts" / "inference.py",
                ROOT / "shared" / "data_utils.py", Path(__file__).resolve()]
    for name in MODELS:
        cfg, ckpt = model_paths(name)
        metrics = ROOT / "models" / name / "results" / "inference" / "inference_metrics.json"
        target_path = metrics.with_name("target_performance.npy")
        target_errors[name] = float(np.max(np.abs(np.load(target_path) - target)))
        assert np.allclose(np.load(target_path), target, rtol=1e-6, atol=1e-6), name
        write_json(out / "inputs" / f"hpo_best_{name}.json", read_json(cfg))
        write_json(out / "inputs" / f"historical_accuracy_{name}.json", read_json(metrics))
        sources += [cfg, ckpt, metrics, target_path, ROOT / "models" / name / "model.py"]
    manifest = {str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p) for p in sources}
    write_json(out / "inputs" / "source_sha256.json", manifest)
    write_json(out / "protocol.json", {
        "started_local": datetime.now().isoformat(), "gpu_index": args.gpu,
        "gpu_before": gpu_status(), "models": MODELS,
        "training_repetitions": "10 consecutive measured epochs following 2 warmup epochs; timing variability, not independent model training repeats.",
        "inference_repetitions": "3 complete 100-condition workloads, sampling seeds 42/43/44.",
        "historical_target_max_abs_difference": target_errors,
        "precision": "FP32, TF32 disabled, no AMP, no torch.compile", "cpu_threads": 4,
        "scope": "Configuration-specific cost profiling; not total historical training/HPO cost or intrinsic algorithm complexity.",
    })
    for name in MODELS:
        for phase in ("inference", "train"):
            command = [sys.executable, "-X", "utf8", "-u", str(Path(__file__).resolve()),
                       "--worker", "--model", name, "--phase", phase,
                       "--gpu", str(args.gpu), "--output", str(out)]
            env = dict(os.environ, PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
            print(f"Starting {name} {phase}", flush=True)
            with (out / "logs" / f"{name}_{phase}.log").open("w", encoding="utf-8") as log:
                proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, text=True, encoding="utf-8")
                for line in proc.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end="", flush=True)
                if proc.wait() != 0:
                    raise RuntimeError(f"Failed {name} {phase}; see log")
    checks = {rel: sha256(ROOT / rel) == digest for rel, digest in manifest.items()}
    write_json(out / "verification.json", {"all_source_files_unchanged": all(checks.values()),
               "source_checks": checks, "gpu_after": gpu_status(), "completed_local": datetime.now().isoformat()})
    assert all(checks.values())
    print(f"Completed. All {len(checks)} source files unchanged. {out}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(HERE / "results" / "rtx4090_fp32"))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--phase", choices=["train", "inference"])
    options = parser.parse_args()
    if options.worker:
        worker(options)
    else:
        main(options)
