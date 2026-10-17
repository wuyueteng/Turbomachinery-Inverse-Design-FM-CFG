"""Time repeated sample(condition, n_samples=10) calls individually after warmup.

Training records are reused verbatim from the completed epoch benchmark.
All inference models run sequentially in separate processes on one GPU.
"""
from pathlib import Path
import argparse
import csv
from datetime import datetime
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time

from run_benchmark import (ROOT, HERE, MODELS, DISPLAY, configure, gpu_status,
                           model_paths, read_json, write_json, sha256)


def csv_file(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def worker(args):
    import numpy as np
    import torch
    import scripts.optuna_hpo as hpo

    out = args.output.resolve()
    target = out / "raw" / f"{args.model}_inference.json"
    if target.exists():
        raise FileExistsError(target)
    device = configure(42, args.gpu)
    hpo.DEVICE = device
    cfg_path, ckpt_path = model_paths(args.model)
    cfg = read_json(cfg_path)
    checkpoint_hash = sha256(ckpt_path)
    config_hash = sha256(cfg_path)
    previous_train = read_json(out / "raw" / f"{args.model}_train.json")
    assert checkpoint_hash == previous_train["checkpoint_sha256"]
    assert config_hash == previous_train["config_sha256"]
    model = hpo.create_model(args.model, cfg["best_params"], device=device)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    del state
    model.to(device).eval()
    if hasattr(model, "temperature"):
        model.temperature = .8
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert parameters == cfg["best_n_params"]
    active_module = model.generator if args.model == "cgan" else model.decoder if args.model == "cvae" else model
    active_parameters = sum(p.numel() for p in active_module.parameters())
    condition = np.load(out / "inputs" / "condition_norm.npy")
    assert condition.shape == (3,)
    result = {
        "model": args.model, "display_name": DISPLAY[args.model], "phase": "inference",
        "started_local": datetime.now().isoformat(),
        "environment": {"platform": platform.platform(), "python": platform.python_version(),
                        "pytorch": torch.__version__, "cuda": torch.version.cuda,
                        "cudnn": torch.backends.cudnn.version(), "gpu_index": args.gpu,
                        "gpu_name": torch.cuda.get_device_name(device), "precision": "float32",
                        "cpu_threads": 4, "matmul_tf32": False, "cudnn_tf32": False,
                        "gpu_before": gpu_status()},
        "checkpoint_sha256": checkpoint_hash, "config_sha256": config_hash,
        "trainable_parameters": parameters, "inference_active_parameters": active_parameters,
        "best_params": cfg["best_params"],
        "memory_definition": "Peak PyTorch allocated tensor memory; full model resident; excludes context, driver and unused allocator cache.",
        "protocol": {"conditions": 1, "designs_per_condition": 10, "total_designs": 10,
                     "calls_per_repetition": 1, "warmup_calls": 5, "measured_repetitions": args.repetitions,
                     "sampling_seeds": list(range(42, 42 + args.repetitions)), "temperature_ar_cinn": .8,
                     "condition_norm": condition.tolist(), "full_model_on_gpu": True,
                     "sampling_implementation_modified": False,
                     "inference_scope": "Each repetition times one original sample(condition, n_samples=10) call after warmup; includes noise, input transfer, clamping and CPU numpy return. Excludes loading, TeM, CFD and file I/O."},
    }
    for _ in range(5):
        warm = model.sample(condition, n_samples=10)
        assert warm.shape == (10, 21) and np.isfinite(warm).all()
        del warm
    measurements, outputs = [], []
    for repetition in range(args.repetitions):
        seed = 42 + repetition
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        generated = model.sample(condition, n_samples=10)
        torch.cuda.synchronize(device)
        seconds = time.perf_counter() - start
        peak = torch.cuda.max_memory_allocated(device)
        reserved = torch.cuda.max_memory_reserved(device)
        assert generated.shape == (10, 21) and np.isfinite(generated).all()
        assert generated.min() >= 0 and generated.max() <= 1
        assert seconds > 0 and reserved >= peak
        outputs.append(generated)
        measurements.append({"repetition": repetition + 1, "sampling_seed": seed, "warmup": False,
                             "seconds": seconds, "peak_allocated_bytes": peak, "peak_reserved_bytes": reserved,
                             "output_shape": list(generated.shape), "all_finite": True,
                             "output_min": float(generated.min()), "output_max": float(generated.max())})
    output_path = out / "outputs" / f"{args.model}_batch10.npy"
    np.save(output_path, np.stack(outputs))
    result["measurements"] = measurements
    result["summary"] = {"seconds_mean": statistics.mean(m["seconds"] for m in measurements),
                         "seconds_sd": statistics.stdev(m["seconds"] for m in measurements),
                         "measurement_count": args.repetitions,
                         "peak_allocated_mib": max(m["peak_allocated_bytes"] for m in measurements) / 2**20,
                         "peak_reserved_mib": max(m["peak_reserved_bytes"] for m in measurements) / 2**20}
    result["output_sha256"] = sha256(output_path)
    result["checkpoint_unchanged"] = sha256(ckpt_path) == checkpoint_hash
    assert result["checkpoint_unchanged"] and sha256(cfg_path) == config_hash
    result["environment"]["gpu_after"] = gpu_status()
    result["finished_local"] = datetime.now().isoformat()
    write_json(target, result)
    print(f'{DISPLAY[args.model]}: {result["summary"]["seconds_mean"]:.6f} s per batch of 10; {args.repetitions} individually timed calls', flush=True)


def summarize(out, source):
    import numpy as np

    previous = {r["code_name"]: r for r in read_json(source / "summary.json")}
    condition = np.load(out / "inputs" / "condition_norm.npy")
    rows, timing_rows = [], []
    for name in MODELS:
        train = read_json(out / "raw" / f"{name}_train.json")
        inf = read_json(out / "raw" / f"{name}_inference.json")
        count = inf["protocol"]["measured_repetitions"]
        assert len(inf["measurements"]) == count >= 2 and inf["protocol"]["total_designs"] == 10
        values = [m["seconds"] for m in inf["measurements"]]
        np.testing.assert_allclose(statistics.mean(values), inf["summary"]["seconds_mean"], rtol=1e-12)
        np.testing.assert_allclose(statistics.stdev(values), inf["summary"]["seconds_sd"], rtol=1e-12)
        np.testing.assert_array_equal(inf["protocol"]["condition_norm"], condition)
        assert inf["environment"]["gpu_index"] == train["environment"]["gpu_index"]
        assert inf["environment"]["gpu_name"] == train["environment"]["gpu_name"]
        output = np.load(out / "outputs" / f"{name}_batch10.npy")
        assert output.shape == (count, 10, 21) and np.isfinite(output).all()
        assert output.min() >= 0 and output.max() <= 1
        assert sha256(out / "outputs" / f"{name}_batch10.npy") == inf["output_sha256"]
        measured = [m for m in train["measurements"] if not m["warmup"]]
        assert len(measured) == 10
        np.testing.assert_allclose(statistics.mean(m["seconds"] for m in measured), previous[name]["train_seconds_per_epoch_mean"])
        np.testing.assert_allclose(statistics.stdev(m["seconds"] for m in measured), previous[name]["train_seconds_per_epoch_sd"])
        assert sha256(out / "raw" / f"{name}_train.json") == sha256(source / "raw" / f"{name}_train.json")
        measurements = inf["measurements"]
        row = {k: v for k, v in previous[name].items() if not k.startswith("inference_")}
        row.update(inference_active_parameters=inf["inference_active_parameters"],
                   inference_seconds_per_10_mean=statistics.mean(values),
                   inference_seconds_per_10_sd=statistics.stdev(values), inference_measurement_count=count,
                   inference_designs_per_call=10,
                   inference_peak_allocated_mib=max(m["peak_allocated_bytes"] for m in measurements) / 2**20,
                   inference_peak_reserved_mib=max(m["peak_reserved_bytes"] for m in measurements) / 2**20)
        assert row["inference_peak_allocated_mib"] >= row["trainable_parameters"] * 4 / 2**20
        rows.append(row)
        for measurement in measurements:
            assert measurement["output_shape"] == [10, 21] and measurement["all_finite"]
            timing_rows.append({"model": row["model"], "repetition": measurement["repetition"],
                               "sampling_seed": measurement["sampling_seed"],
                               "seconds_per_call_of_10": measurement["seconds"],
                               "peak_allocated_mib": measurement["peak_allocated_bytes"] / 2**20})
    write_json(out / "summary.json", rows)
    csv_file(out / "tables" / "computational_cost.csv", rows)
    csv_file(out / "tables" / "inference_batch10_measurements.csv", timing_rows)
    parameter_rows = [{"Model": row["model"], "Parameters (M)": f'{row["parameters_million"]:.3f}',
                       "Exact trainable parameters": row["trainable_parameters"]} for row in rows]
    csv_file(out / "tables" / "table5_parameter_column.csv", parameter_rows)
    table = ["| Model | Training time (s/epoch) | Inference time (s/10 designs) | Peak training memory (MiB) | Peak inference memory (MiB) |",
             "|---|---:|---:|---:|---:|"]
    for row in rows:
        table.append(f'| {row["model"]} | {row["train_seconds_per_epoch_mean"]:.3f} +/- {row["train_seconds_per_epoch_sd"]:.3f} | '
                     f'{row["inference_seconds_per_10_mean"]:.6f} +/- {row["inference_seconds_per_10_sd"]:.6f} | '
                     f'{row["train_peak_allocated_mib"]:.1f} | {row["inference_peak_allocated_mib"]:.1f} |')
    (out / "tables" / "computational_cost.md").write_text("\n".join(table) + "\n", encoding="utf-8")
    fm, ddpm = rows[0], rows[4]
    write_json(out / "comparison_fm_vs_ddpm.json", {
        "relative_reduction_percent": {key: 100 * (1 - fm[key] / ddpm[key]) for key in
                                       ["parameters_million", "train_seconds_per_epoch_mean", "inference_seconds_per_10_mean"]},
        "inference_speedup": ddpm["inference_seconds_per_10_mean"] / fm["inference_seconds_per_10_mean"],
        "inference_measurements_per_model": count})
    write_json(out / "summary_verification.json", {"passed": True, "measured_inference_calls_per_model": count,
               "inference_designs_per_call": 10, "same_condition_for_all_models": True,
               "training_records_reused_byte_for_byte": True, "all_output_arrays_checked": True,
               "inference_means_and_sample_sd_recomputed": True})
    print("\n".join(table), flush=True)


def main(args):
    import numpy as np

    out, source = args.output.resolve(), args.source.resolve()
    if out.exists():
        raise FileExistsError(f"Use a new output directory: {out}")
    assert read_json(source / "summary_verification.json")["passed"]
    for name in ["inputs", "raw", "outputs", "logs", "tables"]:
        (out / name).mkdir(parents=True, exist_ok=False)
    original_conditions = np.load(source / "inputs" / "conditions_norm.npy")
    condition = original_conditions[0]
    np.save(out / "inputs" / "condition_norm.npy", condition)
    test_index = int(np.load(source / "inputs" / "test_condition_indices.npy")[0])
    sources = [Path(__file__).resolve(), HERE / "run_benchmark.py", source / "summary.json",
               source / "inputs" / "conditions_norm.npy", source / "inputs" / "test_condition_indices.npy",
               ROOT / "scripts" / "optuna_hpo.py", ROOT / "shared" / "data_utils.py"]
    for name in MODELS:
        cfg, ckpt = model_paths(name)
        for filename in [f"hpo_best_{name}.json", f"historical_accuracy_{name}.json"]:
            shutil.copy2(source / "inputs" / filename, out / "inputs" / filename)
        training_path = source / "raw" / f"{name}_train.json"
        shutil.copy2(training_path, out / "raw" / training_path.name)
        sources += [cfg, ckpt, ROOT / "models" / name / "model.py", training_path]
    manifest = {p.relative_to(ROOT).as_posix(): sha256(p) for p in sources}
    write_json(out / "inputs" / "source_sha256.json", manifest)
    write_json(out / "protocol.json", {
        "started_local": datetime.now().isoformat(), "gpu_index": args.gpu, "gpu_before": gpu_status(),
        "models": MODELS, "training_source": source.relative_to(ROOT).as_posix(),
        "condition_selection": "First target in the previously saved fixed test-condition list; selected before timing any model.",
        "source_condition_row": 0, "test_split_index": test_index, "condition_norm": condition.tolist(),
        "conditions": 1, "designs_per_call": 10, "warmup_calls": 5, "measured_calls_per_model": args.repetitions,
        "sampling_seeds": list(range(42, 42 + args.repetitions)), "training_remeasured": False,
        "precision": "FP32, TF32 disabled, no AMP, no torch.compile", "cpu_threads": 4,
        "scope": "Mean and sample SD of individually timed warmed calls at a fixed condition; each call generates 10 designs. Not total runtime over all repetitions or an average over target conditions."})
    for name in MODELS:
        command = [sys.executable, "-X", "utf8", "-u", str(Path(__file__).resolve()), "--worker",
                   "--model", name, "--gpu", str(args.gpu), "--output", str(out),
                   "--repetitions", str(args.repetitions)]
        env = dict(os.environ, PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
        print(f"Measuring {args.repetitions} separate batches of 10: {name}", flush=True)
        with (out / "logs" / f"{name}_inference.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            if process.wait() != 0:
                raise RuntimeError(f"Failed {name}; see {log.name}")
        print(read_json(out / "raw" / f"{name}_inference.json")["summary"], flush=True)
    checks = {name: sha256(ROOT / name) == digest for name, digest in manifest.items()}
    assert all(checks.values())
    write_json(out / "verification.json", {"all_source_files_unchanged": True, "source_checks": checks,
               "completed_local": datetime.now().isoformat(), "gpu_after": gpu_status()})
    summarize(out, source)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=HERE / "results" / "rtx4090_fp32")
    parser.add_argument("--output", type=Path, default=HERE / "results" / "rtx4090_fp32_batch10")
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--model", choices=MODELS)
    args = parser.parse_args()
    if args.repetitions < 2:
        parser.error("At least two repetitions are required to estimate timing SD.")
    worker(args) if args.worker else main(args)
