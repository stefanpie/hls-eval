import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator

DIR_CURRENT = Path(__file__).resolve().parent
DIR_FIGURES = DIR_CURRENT / "figures"

DEFAULT_OUTPUT_DATA_DIR = DIR_CURRENT / "output_data"
DEFAULT_PLOT_PATH = DIR_FIGURES / "kernel_latency_mae.png"
DEFAULT_VERTICAL_PLOT_PATH = DIR_FIGURES / "kernel_latency_mae_vertical.png"
DEFAULT_RATIO_PLOT_PATH = DIR_FIGURES / "kernel_latency_ratios.png"

MODELS_TO_PLOT = ["deepseek/deepseek-v4-flash", "openai/gpt-oss-120b"]


def load_kernel_maes(
    output_data_dir: Path,
    model_name: str,
) -> list[tuple[str, list[float], int]]:
    kernel_maes = []

    for results_path in sorted(output_data_dir.glob("*/all_eval_data.json")):
        all_results = json.loads(results_path.read_text())
        model_results = [
            sample
            for sample in all_results.values()
            if sample.get("model_name") == model_name
        ]
        if not model_results:
            continue

        absolute_errors = []
        kernel_name = None

        for sample in model_results:
            kernel_name = sample["benchmark_case_name"]
            estimated_cycles = sample.get("estimated_latency_cycles")
            actual_cycles = sample.get("target_actual_latency_cycles__cosim")

            if estimated_cycles is None or actual_cycles is None:
                continue

            absolute_errors.append(abs(estimated_cycles - actual_cycles))

        if kernel_name is None:
            continue

        kernel_maes.append((kernel_name, sorted(absolute_errors), len(model_results)))

    return sorted(
        kernel_maes,
        key=lambda result: (
            bool(result[1]),
            result[1][0] if result[1] else math.inf,
        ),
    )


def plot_kernel_maes(
    kernel_maes: list[tuple[str, list[float], int]],
    plot_path: Path,
    model_name: str,
) -> None:
    if not kernel_maes:
        raise ValueError("No valid kernel results were found")

    kernel_labels = [
        f"{kernel} ({len(absolute_errors)}/{total_samples})"
        for kernel, absolute_errors, total_samples in kernel_maes
    ]
    max_samples = max(
        1,
        max(len(absolute_errors) for _, absolute_errors, _ in kernel_maes),
    )
    kernel_positions = list(range(len(kernel_maes)))
    group_height = 0.82
    bar_height = group_height / max_samples

    figure_height = max(6.0, 0.38 * len(kernel_maes)) * 0.5
    figure, axis = plt.subplots(figsize=(10, figure_height))

    for rank in range(max_samples):
        positions = []
        errors = []
        for kernel_position, (_, absolute_errors, _) in zip(
            kernel_positions, kernel_maes
        ):
            if rank >= len(absolute_errors):
                continue
            positions.append(
                kernel_position
                - bar_height * len(absolute_errors) / 2
                + bar_height * (rank + 0.5)
            )
            errors.append(absolute_errors[rank])

        axis.barh(
            positions,
            [max(error, 1.0) - 1.0 for error in errors],
            left=1.0,
            height=bar_height,
            facecolor=(0.122, 0.467, 0.706, 0.3),
            edgecolor="tab:blue",
            linewidth=0.5,
        )

    axis.set_title(f"HLS Kernel vs. LLM Model Latency Error - {model_name}")
    axis.set_xscale("log")

    all_errors = [
        error
        for _, absolute_errors, _ in kernel_maes
        for error in absolute_errors
        if error > 0
    ]
    if all_errors:
        lowest_error = min(all_errors)
        highest_error = max(all_errors)
        lower_magnitude = 10 ** math.floor(math.log10(lowest_error))
        upper_magnitude = 10 ** math.floor(math.log10(highest_error))
        x_limit_lower = math.floor(lowest_error / lower_magnitude) * lower_magnitude
        x_limit_upper = math.ceil(highest_error / upper_magnitude) * upper_magnitude
        axis.set_xlim(x_limit_lower, x_limit_upper)
    else:
        axis.set_xlim(1.0, 10.0)
    axis.xaxis.set_major_locator(LogLocator(base=10))

    axis.set_xlabel("LLM Modeled Latency Absolute Error (Clock Cycles, Log Scale)")
    # axis.set_ylabel("Kernel (valid estimates / total samples)")
    axis.set_yticks(kernel_positions)
    axis.set_yticklabels(kernel_labels)
    axis.set_ylim(-0.75, len(kernel_positions) - 0.25)
    for kernel_position, (_, absolute_errors, _) in zip(kernel_positions, kernel_maes):
        if not absolute_errors:
            axis.text(
                0.015,
                kernel_position,
                "X",
                transform=axis.get_yaxis_transform(),
                color="red",
                fontsize=13,
                fontweight="bold",
                ha="left",
                va="center",
            )
    axis.grid(axis="x", which="both", linestyle="--", alpha=0.35)
    axis.set_axisbelow(True)

    figure.tight_layout()
    figure.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_kernel_maes_vertical(
    kernel_maes: list[tuple[str, list[float], int]],
    plot_path: Path,
    model_name: str,
) -> None:
    if not kernel_maes:
        raise ValueError("No valid kernel results were found")

    kernel_labels = [
        f"{kernel} ({len(absolute_errors)}/{total_samples})"
        for kernel, absolute_errors, total_samples in kernel_maes
    ]
    max_samples = max(
        1,
        max(len(absolute_errors) for _, absolute_errors, _ in kernel_maes),
    )
    kernel_positions = list(range(len(kernel_maes)))
    group_width = 0.82
    bar_width = group_width / max_samples

    figure_width = max(10.0, 0.38 * len(kernel_maes)) * 0.8
    figure, axis = plt.subplots(figsize=(figure_width, 3.5))

    for rank in range(max_samples):
        positions = []
        errors = []
        for kernel_position, (_, absolute_errors, _) in zip(
            kernel_positions, kernel_maes
        ):
            if rank >= len(absolute_errors):
                continue
            positions.append(
                kernel_position
                - bar_width * len(absolute_errors) / 2
                + bar_width * (rank + 0.5)
            )
            errors.append(absolute_errors[rank])

        axis.bar(
            positions,
            [max(error, 1.0) - 1.0 for error in errors],
            bottom=1.0,
            width=bar_width,
            facecolor=(0.122, 0.467, 0.706, 0.3),
            edgecolor="tab:blue",
            linewidth=0.5,
        )

    axis.set_title(f"HLS Kernel vs. LLM Model Latency Error - {model_name}")
    axis.set_yscale("log")

    all_errors = [
        error
        for _, absolute_errors, _ in kernel_maes
        for error in absolute_errors
        if error > 0
    ]
    if all_errors:
        lowest_error = min(all_errors)
        highest_error = max(all_errors)
        lower_magnitude = 10 ** math.floor(math.log10(lowest_error))
        upper_magnitude = 10 ** math.floor(math.log10(highest_error))
        y_limit_lower = math.floor(lowest_error / lower_magnitude) * lower_magnitude
        y_limit_upper = math.ceil(highest_error / upper_magnitude) * upper_magnitude
        axis.set_ylim(y_limit_lower, y_limit_upper)
    else:
        axis.set_ylim(1.0, 10.0)
    axis.yaxis.set_major_locator(LogLocator(base=10))

    axis.set_ylabel("LLM Modeled Latency Absolute Error\n(Clock Cycles)")
    axis.set_xticks(kernel_positions)
    axis.set_xticklabels(kernel_labels, rotation=35, ha="right")
    axis.set_xlim(-0.75, len(kernel_positions) - 0.25)
    for kernel_position, (_, absolute_errors, _) in zip(kernel_positions, kernel_maes):
        if not absolute_errors:
            axis.text(
                kernel_position,
                0.015,
                "X",
                transform=axis.get_xaxis_transform(),
                color="red",
                fontsize=13,
                fontweight="bold",
                ha="center",
                va="bottom",
            )
    axis.grid(axis="y", which="both", linestyle="--", alpha=0.35)
    axis.set_axisbelow(True)

    figure.tight_layout()
    figure.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def load_kernel_sample_ratios(
    output_data_dir: Path,
    model_name: str,
) -> list[tuple[str, list[float]]]:
    kernel_ratios = []

    for results_path in sorted(output_data_dir.glob("*/all_eval_data.json")):
        all_results = json.loads(results_path.read_text())
        samples = sorted(
            (
                item
                for item in all_results.items()
                if item[1].get("model_name") == model_name
            ),
            key=lambda item: int(item[0]),
        )
        if not samples:
            continue

        kernel_name = samples[0][1]["benchmark_case_name"]
        sample_ratios = []
        for _, sample in samples:
            estimated_cycles = sample.get("estimated_latency_cycles")
            actual_cycles = sample.get("target_actual_latency_cycles__cosim")
            if (
                estimated_cycles is None
                or actual_cycles is None
                or estimated_cycles <= 0
                or actual_cycles <= 0
            ):
                continue
            sample_ratios.append(estimated_cycles / actual_cycles)

        kernel_ratios.append((kernel_name, sorted(sample_ratios)))

    return sorted(
        kernel_ratios,
        key=lambda result: (
            not result[1],
            sum(result[1]) / len(result[1]) if result[1] else math.inf,
        ),
    )


def plot_kernel_sample_ratios(
    kernel_ratios: list[tuple[str, list[float]]],
    plot_path: Path,
    model_name: str,
) -> None:
    if not kernel_ratios:
        raise ValueError("No kernel results were found")

    max_samples = max(
        1,
        max(len(sample_ratios) for _, sample_ratios in kernel_ratios),
    )
    group_spacing = 1.35
    kernel_positions = [
        kernel_index * group_spacing for kernel_index in range(len(kernel_ratios))
    ]
    group_width = 0.82
    bar_width = group_width / max_samples

    figure_width = max(10.0, 0.38 * len(kernel_ratios)) * 0.8
    figure, axis = plt.subplots(figsize=(figure_width, 4))

    for rank in range(max_samples):
        positions = []
        ratios = []
        for kernel_position, (_, sample_ratios) in zip(kernel_positions, kernel_ratios):
            if rank >= len(sample_ratios):
                continue
            ratio = sample_ratios[rank]
            positions.append(
                kernel_position
                - bar_width * len(sample_ratios) / 2
                + bar_width * (rank + 0.5)
            )
            ratios.append(ratio)

        axis.bar(
            positions,
            [ratio - 1.0 for ratio in ratios],
            width=bar_width,
            bottom=1.0,
            label=f"Rank {rank + 1}",
            facecolor=(0.122, 0.467, 0.706, 0.3),
            edgecolor="tab:blue",
            linewidth=0.5,
        )

    axis.set_yscale("log", base=10)

    axis.axhline(1.0, color="black", linewidth=1.2)
    axis.text(
        kernel_positions[-1] + group_width / 2,
        1.0,
        "1x truth",
        va="bottom",
        ha="right",
        fontweight="bold",
    )

    axis.set_xticks(kernel_positions)
    axis.set_xticklabels(
        [kernel_name for kernel_name, _ in kernel_ratios],
        rotation=55,
        ha="right",
    )
    axis.set_xlim(
        kernel_positions[0] - group_spacing * 0.65,
        kernel_positions[-1] + group_spacing * 0.65,
    )
    for kernel_position, (_, sample_ratios) in zip(kernel_positions, kernel_ratios):
        if not sample_ratios:
            axis.text(
                kernel_position,
                0.025,
                "X",
                transform=axis.get_xaxis_transform(),
                color="red",
                fontsize=13,
                fontweight="bold",
                ha="center",
                va="bottom",
            )
    axis.set_title(f"LLM Modeled Latency Relative to True Latency - {model_name}")
    axis.set_ylabel("LLM Modeled Latency / True Latency Ratio")
    axis.grid(axis="y", which="both", linestyle="--", alpha=0.5)
    axis.set_axisbelow(True)

    figure.tight_layout()
    figure.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def model_plot_path(base_path: Path, model_name: str) -> Path:
    normalized_model_name = (
        model_name.replace("/", "_").replace("-", "_").replace(" ", "_").lower()
    )
    return base_path.with_name(
        f"{base_path.stem}__{normalized_model_name}{base_path.suffix}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot per-kernel MAE between estimated and synthesized HLS latency."
        )
    )
    parser.add_argument(
        "--output-data-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DATA_DIR,
        help="Directory containing per-kernel evaluation result directories",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_PLOT_PATH,
        help="Path for the generated horizontal-bar MAE PNG plot",
    )
    parser.add_argument(
        "--vertical-output",
        type=Path,
        default=DEFAULT_VERTICAL_PLOT_PATH,
        help="Path for the generated vertical-bar MAE PNG plot",
    )
    parser.add_argument(
        "--ratio-output",
        type=Path,
        default=DEFAULT_RATIO_PLOT_PATH,
        help="Path for the generated per-sample latency ratio PNG plot",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.vertical_output.parent.mkdir(parents=True, exist_ok=True)
    args.ratio_output.parent.mkdir(parents=True, exist_ok=True)

    for model_name in MODELS_TO_PLOT:
        mae_plot_path = model_plot_path(args.output, model_name)
        kernel_maes = load_kernel_maes(args.output_data_dir, model_name)
        if kernel_maes:
            plot_kernel_maes(kernel_maes, mae_plot_path, model_name)
            print(f"Saved {model_name} plot to {mae_plot_path}")

            vertical_mae_plot_path = model_plot_path(args.vertical_output, model_name)
            plot_kernel_maes_vertical(kernel_maes, vertical_mae_plot_path, model_name)
            print(f"Saved {model_name} vertical plot to {vertical_mae_plot_path}")
        else:
            print(f"Skipping {model_name} MAE plot: no valid model results")

        ratio_plot_path = model_plot_path(args.ratio_output, model_name)
        kernel_ratios = load_kernel_sample_ratios(args.output_data_dir, model_name)
        if kernel_ratios:
            plot_kernel_sample_ratios(kernel_ratios, ratio_plot_path, model_name)
            print(f"Saved {model_name} ratio plot to {ratio_plot_path}")
        else:
            print(f"Skipping {model_name} ratio plot: no valid model results")


if __name__ == "__main__":
    main()
