"""Paper-style cost figure and Table 5 parameter column, using verified results only."""
from pathlib import Path
import argparse
import ast
import csv
import hashlib
import json
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, LogLocator
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ORDER = ["fm_cfg", "ar", "cvae", "cgan", "ddpm", "cinn"]
STYLE_SOURCE = ROOT / "scripts" / "plot_bubble.py"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_file(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def original_palette():
    # Read constants without importing the historical plotting script or its side effects.
    tree = ast.parse(STYLE_SOURCE.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "COLORS" for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError("Original manuscript model palette was not found")


def verify_rows(folder):
    rows = read(folder / "summary.json")
    assert [r["code_name"] for r in rows] == ORDER
    assert read(folder / "summary_verification.json")["passed"]
    sources = [folder / "summary.json", STYLE_SOURCE]
    for row in rows:
        name = row["code_name"]
        cfg_path = folder / "inputs" / f"hpo_best_{name}.json"
        assert read(cfg_path)["best_n_params"] == row["trainable_parameters"]
        sources.append(cfg_path)
        batch10 = "inference_seconds_per_10_mean" in row
        for phase, count in [("train", 10), ("inference", row["inference_measurement_count"] if batch10 else 3)]:
            path = folder / "raw" / f"{name}_{phase}.json"
            record = read(path)
            measured = [m for m in record["measurements"] if not m["warmup"]]
            assert len(measured) == count
            key = "train_seconds_per_epoch" if phase == "train" else ("inference_seconds_per_10" if batch10 else "inference_seconds_per_1000")
            if phase == "inference" and batch10:
                assert record["protocol"]["total_designs"] == 10
                assert record["protocol"]["calls_per_repetition"] == 1
                assert all(m["output_shape"] == [10, 21] and m["all_finite"] for m in measured)
            np.testing.assert_allclose(row[key + "_mean"], statistics.mean(m["seconds"] for m in measured), rtol=1e-12)
            np.testing.assert_allclose(row[key + "_sd"], statistics.stdev(m["seconds"] for m in measured), rtol=1e-12)
            expected_peak = max(m["peak_allocated_bytes"] for m in measured) / 2**20
            assert expected_peak == row[f"{phase}_peak_allocated_mib"]
            sources.append(path)
    return rows, {str(p.relative_to(ROOT)).replace("\\", "/"): digest(p) for p in sources}


def style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix", "axes.linewidth": 1.0,
        "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True,
        "ytick.right": True, "xtick.major.size": 5, "ytick.major.size": 4,
        "xtick.major.width": .9, "ytick.major.width": .9,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "savefig.facecolor": "white", "figure.facecolor": "white",
    })


def panel(ax, kind, rows, palette, standalone=False):
    label_fs, tick_fs, value_fs = (24, 18, 16) if standalone else (17, 15, 13)
    positions = np.arange(len(rows))
    ax.set_ylim(len(rows) - .4, -.85)
    ax.set_yticks(positions, labels=[r["model"] for r in rows])
    ax.tick_params(axis="both", labelsize=tick_fs)
    ax.tick_params(axis="y", length=0, pad=9)
    ax.grid(axis="y", linestyle="--", color="#999999", alpha=.24, linewidth=.55)
    ax.set_axisbelow(True)
    handles = []
    labels = []
    if kind == "training":
        ax.set_xlim(0, .91)
        ax.set_xticks([0, .2, .4, .6, .8])
        ax.set_xlabel("Training time (s/epoch)", fontsize=label_fs, labelpad=9)
        key = "train_seconds_per_epoch"
    elif kind == "inference":
        ax.set_xscale("log")
        ax.set_xlim(.045, 360)
        ax.xaxis.set_major_locator(FixedLocator([.1, 1, 10, 100]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
        ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10)))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("Inference time (s/1,000 designs)", fontsize=label_fs, labelpad=9)
        key = "inference_seconds_per_1000"
    else:
        ax.set_xscale("log")
        ax.set_xlim(12, 3500)
        ax.xaxis.set_major_locator(FixedLocator([20, 100, 500, 2000]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("Peak GPU memory (MiB)", fontsize=label_fs, labelpad=9)
        handles = [Line2D([], [], marker="s", color="#333333", linestyle="none", markersize=7),
                   Line2D([], [], marker="o", color="#333333", markerfacecolor="white", linestyle="none", markersize=7)]
        labels = ["Training", "Inference"]
        ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(.53, 1.12),
                  ncol=2, frameon=False, fontsize=tick_fs - 1, handletextpad=.3, columnspacing=1.0)
    for y, row in zip(positions, rows):
        color = palette[row["code_name"]]
        if kind != "memory":
            mean, sd = row[key + "_mean"], row[key + "_sd"]
            ax.errorbar(mean, y, xerr=sd, fmt="o", color=color, ecolor=color,
                        markersize=8, elinewidth=1.5, capsize=4, markeredgecolor="white",
                        markeredgewidth=.65, zorder=4)
            ax.annotate(f"{mean:.3f}", (mean, y), xytext=(0, 9),
                        textcoords="offset points", ha="center", va="bottom", fontsize=value_fs)
        else:
            tr = row["train_peak_allocated_mib"]
            inf = row["inference_peak_allocated_mib"]
            ax.plot([inf, tr], [y, y], color=color, alpha=.45, lw=1.5, zorder=2)
            ax.plot(tr, y, marker="s", color=color, markersize=7.5, linestyle="none", zorder=4)
            ax.plot(inf, y, marker="o", markeredgecolor=color, markerfacecolor="white",
                    markeredgewidth=1.6, markersize=7.5, linestyle="none", zorder=4)
            for value in [inf, tr]:
                ax.annotate(f"{value:.1f}", (value, y), xytext=(0, 9),
                            textcoords="offset points", ha="center", va="bottom", fontsize=value_fs)
    return handles, labels


def save(fig, folder, name, audit):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width, height = fig.canvas.get_width_height()
    clipped = []
    for ax in fig.axes:
        texts = [ax.xaxis.label, ax.yaxis.label, *ax.get_xticklabels(), *ax.get_yticklabels(), *ax.texts]
        for item in texts:
            if not item.get_visible() or not item.get_text():
                continue
            box = item.get_window_extent(renderer)
            if box.x0 < -1 or box.y0 < -1 or box.x1 > width + 1 or box.y1 > height + 1:
                clipped.append(item.get_text())
    assert not clipped, (name, clipped)
    files = []
    for ext in ["png", "pdf", "svg"]:
        target = folder / f"{name}.{ext}"
        fig.savefig(target, dpi=600)
        files.append({"path": target.name, "sha256": digest(target)})
    audit[name] = {"all_text_inside_canvas": True, "files": files}
    plt.close(fig)


def main(folder):
    rows, source_hashes = verify_rows(folder)
    figures = HERE / "figures"
    figures.mkdir(exist_ok=True)
    tables = folder / "tables"
    tables.mkdir(exist_ok=True)
    style()
    palette = original_palette()
    audit = {}
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.1), sharey=True)
    fig.subplots_adjust(left=.087, right=.985, top=.87, bottom=.16, wspace=.20)
    for ax, kind, letter in zip(axes, ["training", "inference", "memory"], ["(a)", "(b)", "(c)"]):
        panel(ax, kind, rows, palette)
        ax.text(0, 1.09, letter, transform=ax.transAxes, fontsize=18, va="bottom", ha="left")
    for ax in axes[1:]:
        ax.tick_params(axis="y", labelleft=False)
    save(fig, figures, "computational_cost_comparison", audit)
    for kind in ["training", "inference", "memory"]:
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        fig.subplots_adjust(left=.235, right=.97, top=.88, bottom=.16)
        panel(ax, kind, rows, palette, standalone=True)
        save(fig, figures, f"panel_{kind}", audit)
    parameter_rows = [{"Model": r["model"], "Parameters (M)": f'{r["parameters_million"]:.3f}',
                       "Exact trainable parameters": r["trainable_parameters"]} for r in rows]
    csv_file(tables / "table5_parameter_column.csv", parameter_rows)
    table = ["| Model | Parameters (M) |", "|---|---:|"]
    table += [f'| {r["model"]} | {r["parameters_million"]:.3f} |' for r in rows]
    (tables / "table5_parameter_column.md").write_text("\n".join(table) + "\n", encoding="utf-8")
    csv_file(tables / "figure_plot_data.csv", rows)
    unchanged = {p: digest(ROOT / p) == h for p, h in source_hashes.items()}
    assert all(unchanged.values())
    (folder / "figure_verification.json").write_text(json.dumps({
        "source_hashes": source_hashes, "sources_unchanged": unchanged,
        "values_recomputed_from_raw_records": True, "order_matches_table5": ORDER,
        "error_bars": "sample SD; 10 measured epochs / 3 full inference workloads",
        "scales": {"training": "linear", "inference": "log10", "memory": "log10"},
        "memory": "maximum allocated, not mean and no SD",
        "style_source": str(STYLE_SOURCE), "outputs": audit,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Verified and saved combined figure and three panels in {figures}")
    print("\n".join(table))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=HERE / "results" / "rtx4090_fp32")
    main(parser.parse_args().results)
