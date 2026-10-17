"""Validate timing records and prepare the computational-cost table."""
from pathlib import Path
import argparse
import csv
import json
import math
import statistics
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from run_benchmark import MODELS, DISPLAY, read_json, write_json, sha256


def save_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value, sd=None):
    if sd is None:
        return f"{value:.3f}"
    digits = max(3, min(6, math.ceil(-math.log10(sd)))) if sd > 0 else 3
    return f"{value:.{digits}f} ± {sd:.{digits}f}"


def main(folder):
    import numpy as np
    folder = folder.resolve()
    tables = folder / "tables"
    tables.mkdir(exist_ok=True)
    source_manifest = read_json(folder / "inputs" / "source_sha256.json")
    source_checks = {key: sha256(ROOT / key) == value for key, value in source_manifest.items()}
    assert all(source_checks.values())
    assert read_json(folder / "verification.json")["all_source_files_unchanged"]
    expected_gpu = read_json(folder / "protocol.json")["gpu_index"]
    conditions = np.load(folder / "inputs" / "conditions_norm.npy")
    assert conditions.shape == (100, 3) and np.isfinite(conditions).all()
    rows, time_rows, checks = [], [], {}
    for name in MODELS:
        train = read_json(folder / "raw" / f"{name}_train.json")
        inf = read_json(folder / "raw" / f"{name}_inference.json")
        accuracy = read_json(folder / "inputs" / f"historical_accuracy_{name}.json")
        assert train["checkpoint_sha256"] == inf["checkpoint_sha256"]
        assert train["config_sha256"] == inf["config_sha256"]
        assert train["checkpoint_unchanged"] and inf["checkpoint_unchanged"]
        assert train["trainable_parameters"] == inf["trainable_parameters"]
        assert train["protocol"]["batches_per_epoch"] == 43
        assert train["protocol"]["processed_samples_per_epoch"] == 11008
        assert train["protocol"]["train_split_size"] == 11072
        assert inf["protocol"]["total_designs"] == 1000
        assert train["environment"]["precision"] == inf["environment"]["precision"] == "float32"
        assert train["environment"]["gpu_index"] == inf["environment"]["gpu_index"] == expected_gpu
        for phase, record, count in [("train", train, 10), ("inference", inf, 3)]:
            measured = [r for r in record["measurements"] if not r["warmup"]]
            assert len(measured) == count
            values = [r["seconds"] for r in measured]
            assert all(v > 0 for v in values)
            assert abs(statistics.mean(values) - record["summary"]["seconds_mean"]) < 1e-10
            assert abs(statistics.stdev(values) - record["summary"]["seconds_sd"]) < 1e-10
            peak = max(r["peak_allocated_bytes"] for r in measured) / 2**20
            assert peak == record["summary"]["peak_allocated_mib"]
            assert peak >= record["trainable_parameters"] * 4 / 2**20
            for j, m in enumerate(record["measurements"]):
                assert m["peak_reserved_bytes"] >= m["peak_allocated_bytes"]
                if phase == "inference":
                    assert m["all_finite"] and m["output_shape"] == [1000, 21]
                else:
                    assert np.isfinite(m["loss"])
                    if name == "cgan":
                        assert m["generator_updates"] + m["discriminator_updates"] == 43
                time_rows.append({"model": DISPLAY[name], "phase": phase, "index": j + 1,
                                  "warmup": m["warmup"], "seconds": m["seconds"],
                                  "peak_allocated_mib": m["peak_allocated_bytes"] / 2**20,
                                  "peak_reserved_mib": m["peak_reserved_bytes"] / 2**20,
                                  "loss": m.get("loss", ""),
                                  "generator_updates": m.get("generator_updates", ""),
                                  "discriminator_updates": m.get("discriminator_updates", "")})
        ts, ins = train["summary"], inf["summary"]
        row = {
            "model": DISPLAY[name], "code_name": name,
            "mean_nrmse_existing": statistics.mean(accuracy[f"nrmse_{k}"] for k in ["flow", "eff", "pr"]),
            "trainable_parameters": train["trainable_parameters"],
            "parameters_million": train["trainable_parameters"] / 1e6,
            "inference_active_parameters": inf["inference_active_parameters"],
            "train_seconds_per_epoch_mean": ts["seconds_mean"],
            "train_seconds_per_epoch_sd": ts["seconds_sd"],
            "inference_seconds_per_1000_mean": ins["seconds_mean"],
            "inference_seconds_per_1000_sd": ins["seconds_sd"],
            "inference_ms_per_design_amortized": ins["seconds_mean"],
            "train_peak_allocated_mib": ts["peak_allocated_mib"],
            "inference_peak_allocated_mib": ins["peak_allocated_mib"],
            "train_peak_reserved_mib": ts["peak_reserved_mib"],
            "inference_peak_reserved_mib": ins["peak_reserved_mib"],
        }
        rows.append(row)
        checks[name] = {"parameter_counts_verified": True, "training_epochs_verified": 10,
                        "inference_repetitions_verified": 3, "output_shape_and_finiteness": True,
                        "checkpoint_unchanged": True, "raw_statistics_recomputed": True}
    save_csv(tables / "computational_cost.csv", rows)
    save_csv(tables / "timing_measurements.csv", time_rows)
    write_json(folder / "summary.json", rows)
    write_json(folder / "summary_verification.json", {
        "passed": True, "source_files_unchanged": source_checks,
        "checks": checks, "timing_records": len(time_rows),
        "historical_accuracy_reused_not_retrained": True})

    header = "| Model | Existing mean nRMSE | Parameters (M) | Training (s/epoch) | Inference (s/1,000 designs) | Peak training memory (MiB) | Peak inference memory (MiB) |"
    table = [header, "|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        table.append("| " + " | ".join([
            row["model"], f'{row["mean_nrmse_existing"]:.4f}', fmt(row["parameters_million"]),
            fmt(row["train_seconds_per_epoch_mean"], row["train_seconds_per_epoch_sd"]),
            fmt(row["inference_seconds_per_1000_mean"], row["inference_seconds_per_1000_sd"]),
            f'{row["train_peak_allocated_mib"]:.1f}', f'{row["inference_peak_allocated_mib"]:.1f}'
        ]) + " |")
    table_text = "\n".join(table)
    (tables / "computational_cost.md").write_text(table_text + "\n", encoding="utf-8")
    extended = ["| Metrics | " + " | ".join(DISPLAY[n] for n in MODELS) + " |",
                "|---|---:|---:|---:|---:|---:|---:|"]
    for label, key, sd_key in [
        ("Trainable parameters (M)", "parameters_million", None),
        ("Training time (s/epoch)", "train_seconds_per_epoch_mean", "train_seconds_per_epoch_sd"),
        ("Inference time (s/1,000 designs)", "inference_seconds_per_1000_mean", "inference_seconds_per_1000_sd"),
        ("Peak training GPU memory (MiB)", "train_peak_allocated_mib", None),
        ("Peak inference GPU memory (MiB)", "inference_peak_allocated_mib", None),
    ]:
        values = [f"{r[key]:.1f}" if "memory" in label else
                  fmt(r[key], r[sd_key] if sd_key else None) for r in rows]
        extended.append(f"| {label} | " + " | ".join(values) + " |")
    extended_text = "\n".join(extended)
    (tables / "table6_additional_rows.md").write_text(extended_text + "\n", encoding="utf-8")

    fm = rows[0]
    ddpm = next(r for r in rows if r["code_name"] == "ddpm")
    fastest = min(rows, key=lambda r: r["inference_seconds_per_1000_mean"])
    reduction = {k: 100 * (1 - fm[k] / ddpm[k]) for k in
                 ["parameters_million", "train_seconds_per_epoch_mean", "inference_seconds_per_1000_mean"]}
    write_json(folder / "comparison_fm_vs_ddpm.json", {"relative_reduction_percent": reduction,
               "inference_speedup": ddpm["inference_seconds_per_1000_mean"] / fm["inference_seconds_per_1000_mean"]})
    print(table_text)
    print(f"Verified {len(source_checks)} unchanged sources; {len(time_rows)} raw timing records.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=HERE / "results" / "rtx4090_fp32")
    args = parser.parse_args()
    main(args.results)
