"""One-axis grouped bars for runtime and memory; clean publication view by default.

Time metrics use their respective across-model maxima. Both memory series use
one common maximum, preserving the training/inference memory ratio. No bubble,
secondary axes, minimum-width inflation, or composite cost score is used.
"""
from pathlib import Path
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

from plot_costs import HERE, ROOT, ORDER, verify_rows, original_palette, style, save, digest, csv_file


def main(folder, annotated=False, numbers_only=False, inset=False):
    rows, sources = verify_rows(folder)
    sources[str(Path(__file__).relative_to(ROOT)).replace("\\", "/")] = digest(Path(__file__))
    sources["experiments/computational_cost/plot_costs.py"] = digest(HERE / "plot_costs.py")
    for source in ["scripts/plot_regression.py", "experiments/ablation/plot_cfg_ablation.py"]:
        sources[source] = digest(ROOT / source)
    figures = HERE / "figures"
    figures.mkdir(exist_ok=True)
    table_dir = folder / "tables"
    table_dir.mkdir(exist_ok=True)
    style()
    # Match the regression/ablation figures: boxed axes, inward ticks, Times/STIX.
    plt.rcParams.update({"axes.linewidth": 1.4, "xtick.major.width": 1.0,
                         "ytick.major.width": 1.0, "ytick.major.size": 5})
    palette = original_palette()
    batch10 = "inference_seconds_per_10_mean" in rows[0]
    designs = 10 if batch10 else 1000
    infer_key = f"inference_seconds_per_{designs}"
    inset_threshold = .01 if batch10 else 1.0
    def time_text(value):
        return f"{value:.5f}" if value < .01 else f"{value:.3f}"
    train_ref = max(r["train_seconds_per_epoch_mean"] for r in rows)
    infer_ref = max(r[infer_key + "_mean"] for r in rows)
    memory_ref = max(r[k] for r in rows for k in ["train_peak_allocated_mib", "inference_peak_allocated_mib"])
    specs = [
        {"key": "train_seconds_per_epoch_mean", "sd": "train_seconds_per_epoch_sd",
         "name": "Training time", "unit": "s/epoch", "reference": train_ref,
         "color": palette["fm_cfg"], "hatch": None},
        {"key": infer_key + "_mean", "sd": infer_key + "_sd",
         "name": "Inference time", "unit": f"s/{designs:,} designs", "reference": infer_ref,
         "color": palette["ar"], "hatch": None},
        {"key": "train_peak_allocated_mib", "sd": None,
         "name": "Training memory", "unit": "MiB", "reference": memory_ref,
         "color": palette["cvae"], "hatch": None},
        {"key": "inference_peak_allocated_mib", "sd": None,
         "name": "Inference memory", "unit": "MiB", "reference": memory_ref,
         "color": palette["cvae"], "hatch": "////"},
    ]
    numbered = numbers_only or inset
    wide = annotated or numbered
    fig, ax = plt.subplots(figsize=(11.7, 8.6) if wide else (8.8, 6.9))
    fig.subplots_adjust(left=.16 if wide else .20, right=.96,
                        top=.92, bottom=.115)
    centers = np.arange(len(rows)) * 1.13
    offsets = [-.30, -.10, .10, .30]
    patches, texts, plot_rows, bands = [], [], [], []
    for i, row in enumerate(rows):
        y0 = centers[i]
        if i % 2 == 0:
            bands.append(ax.axhspan(y0 - .48, y0 + .48, color="#F5F6F7",
                                    linewidth=0, clip_on=True, zorder=0))
        for offset, spec in zip(offsets, specs):
            raw = row[spec["key"]]
            relative = 100 * raw / spec["reference"]
            sd = row[spec["sd"]] if spec["sd"] else 0.0
            relative_sd = 100 * sd / spec["reference"]
            y = y0 + offset
            bar = ax.barh(y, relative, height=.145, left=0,
                          color="white" if spec["hatch"] else spec["color"],
                          edgecolor=spec["color"], linewidth=.85, hatch=spec["hatch"],
                          zorder=3)
            patches.append(bar.patches[0])
            if spec["sd"]:
                ax.errorbar(relative, y, xerr=relative_sd, fmt="none", color="#333333",
                            elinewidth=.85, capsize=2.3, capthick=.8, zorder=4)
            number = f"{raw:.1f}" if spec["unit"] == "MiB" else time_text(raw)
            value_text = f"{number} {spec['unit']}"
            if wide:
                # Optional review version. Tiny real bars remain tiny.
                label_x = relative + relative_sd + 1.55
                short_unit = "MiB" if spec["unit"] == "MiB" else "s"
                label = value_text if annotated else (f"{number} {short_unit}" if inset else number)
                text = ax.text(label_x, y, label,
                               va="center", ha="left", fontsize=14.5 if annotated else 13.5,
                               color="#242424", zorder=5)
                texts.append(text)
            plot_rows.append({"model": row["model"], "metric": spec["name"],
                              "source_key": spec["key"], "value": raw, "unit": spec["unit"],
                              "sd": sd if spec["sd"] else "", "normalization_reference": spec["reference"],
                              "relative_percent": relative, "relative_sd_percent": relative_sd,
                              "raw_value_text": value_text})
    ax.set_ylim(centers[-1] + .56, centers[0] - .58)
    # Extend the actual axis and its ticks, not only the plotting rectangle.
    # 100 remains the normalization reference; the margin accommodates SD/labels.
    x_max = 160 if annotated else 120
    ax.set_xlim(0, x_max)
    ax.set_xticks(np.arange(0, x_max + 1, 20))
    ax.set_yticks(centers, labels=[r["model"] for r in rows])
    ax.tick_params(axis="x", labelsize=18 if wide else 15, top=True, direction="in")
    ax.tick_params(axis="y", labelsize=18, length=5, pad=12, right=True, direction="in")
    ax.set_xlabel("Relative cost (% of metric maximum)" if annotated else "Normalized cost (%)",
                  fontsize=24 if wide else 20, labelpad=13)
    ax.grid(axis="x", color="#AAAAAA", linestyle=":", linewidth=.5, alpha=.4, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.4)
    legend_handles = [Patch(facecolor="white" if s["hatch"] else s["color"],
                            edgecolor=s["color"], hatch=s["hatch"], label=s["name"]) for s in specs]
    # One row, ordered as the four bars in each model group.
    legend = ax.legend(handles=legend_handles, ncol=4,
                       loc="lower center", bbox_to_anchor=(.5, 1.025), fontsize=17 if wide else 12,
                       frameon=False, handlelength=1.4, handletextpad=.55, columnspacing=1.15,
                       borderaxespad=0, borderpad=.2)
    inset_ax = None
    inset_records = []
    inset_texts = []
    if inset:
        # This rectangle lies entirely in the empty right-hand part of the cVAE row.
        # It reveals the three very short inference bars without changing their widths.
        inset_ax = ax.inset_axes([.64, .525, .31, .12])
        selected = [r for r in rows if r[infer_key + "_mean"] < inset_threshold]
        assert [r["code_name"] for r in selected] == ["cvae", "cgan", "cinn"]
        for i, row in enumerate(selected):
            value = row[infer_key + "_mean"]
            sd = row[infer_key + "_sd"]
            inset_ax.barh(i, value, height=.48, color=palette["ar"], edgecolor="none", zorder=3)
            inset_ax.errorbar(value, i, xerr=sd, fmt="none", color="#333333", elinewidth=.6, capsize=1.7, zorder=4)
            inset_texts.append(inset_ax.text(value + sd + .012 * inset_threshold, i, f"{time_text(value)} s",
                                            va="center", fontsize=9.5))
            inset_records.append({"model": row["model"], "inference_seconds": value, "sd": sd})
        inset_ax.set_xlim(0, .6 * inset_threshold)
        inset_ax.set_ylim(2.5, -.5)
        inset_ax.set_yticks(range(3), labels=[r["model"] for r in selected])
        inset_ax.set_xticks(np.array([0, .2, .4, .6]) * inset_threshold)
        inset_ax.tick_params(axis="both", labelsize=9.5, length=2.5, width=.7,
                             top=True, right=True, direction="in")
        inset_ax.tick_params(axis="y", pad=4)
        inset_ax.set_title(f"Inference time < {inset_threshold:g} s", fontsize=11, pad=6)
        inset_ax.grid(axis="x", linestyle=":", alpha=.4, linewidth=.5)
        inset_ax.set_axisbelow(True)
        for spine in inset_ax.spines.values():
            spine.set_linewidth(.7)
        inset_ax.set_facecolor("white")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes_box = ax.get_window_extent(renderer)
    def inside(box, boundary, tolerance=.5):
        return (box.x0 >= boundary.x0 - tolerance and box.x1 <= boundary.x1 + tolerance
                and box.y0 >= boundary.y0 - tolerance and box.y1 <= boundary.y1 + tolerance)
    label_boxes = [t.get_window_extent(renderer) for t in texts]
    for i, box in enumerate(label_boxes):
        assert inside(box, axes_box), "Raw value label extends outside the axes"
        for other in label_boxes[i + 1:]:
            assert not box.overlaps(other), "Raw value labels overlap"
    for band in bands:
        box = band.get_window_extent(renderer)
        assert inside(box, axes_box), "Background band extends outside the axes"
        np.testing.assert_allclose([box.x0, box.x1], [axes_box.x0, axes_box.x1], atol=.5)
    for side in ["bottom", "top"]:
        spine = ax.spines[side]
        vertices = spine.get_transform().transform(spine.get_path().vertices)
        np.testing.assert_allclose([vertices[:, 0].min(), vertices[:, 0].max()],
                                   [axes_box.x0, axes_box.x1], atol=.5)
    legend_box = legend.get_window_extent(renderer)
    width, height = fig.canvas.get_width_height()
    assert legend_box.x0 >= 0 and legend_box.x1 <= width and legend_box.y1 <= height
    assert legend_box.x0 >= axes_box.x0 and legend_box.x1 <= axes_box.x1, "Legend exceeds plot width"
    if inset_ax is not None:
        inset_box = inset_ax.get_tightbbox(renderer)
        for box in label_boxes:
            assert not inset_box.overlaps(box), "Inset overlaps a main-plot value label"
        for patch in patches:
            assert not inset_box.overlaps(patch.get_window_extent(renderer)), "Inset overlaps a main-plot bar"
        assert inset_box.x0 >= 0 and inset_box.x1 <= width and inset_box.y0 >= 0 and inset_box.y1 <= height
        for child in [*inset_texts, *inset_ax.get_xticklabels(), *inset_ax.get_yticklabels(), inset_ax.title]:
            box = child.get_window_extent(renderer)
            assert box.x0 >= 0 and box.x1 <= width and box.y0 >= 0 and box.y1 <= height
        for label in inset_texts:
            assert inside(label.get_window_extent(renderer), inset_ax.get_window_extent(renderer))
        for row in inset_records:
            assert 0 <= row["inference_seconds"] - row["sd"]
            assert row["inference_seconds"] + row["sd"] < inset_ax.get_xlim()[1]
    for bar, row in zip(patches, plot_rows):
        np.testing.assert_allclose(bar.get_width(), row["relative_percent"], rtol=1e-12)
        assert bar.get_x() == 0
        assert inside(bar.get_window_extent(renderer), axes_box), "Bar extends outside the axes"
        assert 0 <= row["relative_percent"] - row["relative_sd_percent"]
        assert row["relative_percent"] + row["relative_sd_percent"] < x_max
    audit = {}
    name = ("computational_cost_single_annotated" if annotated else
            "computational_cost_inset" if inset else
            "computational_cost_numbers" if numbers_only else "computational_cost_single")
    save(fig, figures, name, audit)
    csv_file(table_dir / "single_figure_plot_data.csv", plot_rows)
    unchanged = {p: digest(ROOT / p) == h for p, h in sources.items()}
    assert all(unchanged.values())
    report = {
        "axes_count": 2 if inset else 1, "main_bar_count": 24, "bar_count": 24 + len(inset_records), "model_order": ORDER,
        "normalization": {"training_time_s_per_epoch": train_ref,
                          f"inference_time_s_per_{designs}": infer_ref,
                          "shared_training_inference_memory_mib": memory_ref},
        "bar_widths_equal_exact_normalized_values": True,
        "zero_baseline": True, "no_minimum_width_inflation": True,
        "rendered_value_label_count": len(texts),
        "value_labels_contain_units": annotated or inset,
        "rendered_value_labels": [label.get_text() for label in texts],
        "legend": {"rows": 1, "columns": 4, "within_plot_width": True,
                   "labels": [label.get_text() for label in legend.get_texts()]},
        "axis_layout": {"x_limits": list(ax.get_xlim()), "x_ticks": ax.get_xticks().tolist(),
                        "normalization_maximum_unchanged": 100,
                        "all_four_spines_visible": True, "spine_width": 1.4,
                        "tick_direction": "in", "axis_label_fontsize": ax.xaxis.label.get_fontsize(),
                        "axis_line_and_background_have_same_extent": True,
                        "bars_errorbars_and_value_labels_inside_axes": True},
        "inference_protocol": {"designs_per_timed_workload": designs,
                               "measured_repetitions": rows[0].get("inference_measurement_count", 3)},
        "inset": {"records": inset_records, "selection_rule": f"inference time for {designs} designs < {inset_threshold:g} seconds",
                  "units": f"s per {designs} designs", "scale": "linear", "overlaps_main_data": False} if inset else None,
        "all_raw_value_labels_nonoverlapping": True,
        "source_hashes": sources, "sources_unchanged": unchanged,
        "outputs": audit,
    }
    report_name = ("single_figure_annotated_verification.json" if annotated else
                   "inset_figure_verification.json" if inset else
                   "numbers_figure_verification.json" if numbers_only else "single_figure_verification.json")
    (folder / report_name).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved grouped bar figure: {figures / (name + '.png')}")
    print(json.dumps(report["normalization"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=HERE / "results" / "rtx4090_fp32_batch10")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--annotated", action="store_true", help="Export an optional version with values and units.")
    mode.add_argument("--numbers-only", action="store_true", help="Add absolute numbers without repeated units.")
    mode.add_argument("--inset", action="store_true", help="Values with short s/MiB units plus a fast-inference inset.")
    args = parser.parse_args()
    main(args.results, annotated=args.annotated, numbers_only=args.numbers_only, inset=args.inset)
