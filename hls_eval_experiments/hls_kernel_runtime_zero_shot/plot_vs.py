import argparse
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

DIR_CURRENT = Path(__file__).resolve().parent
DIR_FIGURES = DIR_CURRENT / "figures"

DEFAULT_OUTPUT_DATA_DIR = DIR_CURRENT / "output_data"
DEFAULT_PLOT_PATH = DIR_FIGURES / "kernel_latency_vs.png"
DEFAULT_MAE_HIST_PLOT_PATH = DIR_FIGURES / "kernel_latency_mae_hist.png"
DEFAULT_RESIDUAL_PLOT_PATH = DIR_FIGURES / "kernel_latency_residuals.png"
DEFAULT_ABS_ERROR_VS_TRUE_PLOT_PATH = DIR_FIGURES / "kernel_latency_abs_error_vs_true.png"

MODELS_TO_PLOT = ["deepseek/deepseek-v4-flash", "openai/gpt-oss-120b"]


def load_true_vs_predicted(
    output_data_dir: Path,
    model_name: str,
) -> list[tuple[float, float]]:
    true_vs_predicted = []

    for results_path in sorted(output_data_dir.glob("*/all_eval_data.json")):
        all_results = json.loads(results_path.read_text())
        model_results = [
            sample
            for sample in all_results.values()
            if sample.get("model_name") == model_name
        ]

        for sample in model_results:
            estimated_cycles = sample.get("estimated_latency_cycles")
            actual_cycles = sample.get("target_actual_latency_cycles__cosim")

            if estimated_cycles is None or actual_cycles is None:
                continue

            true_vs_predicted.append((actual_cycles, estimated_cycles))

    return true_vs_predicted


def load_kernel_median_true_vs_predicted(
    output_data_dir: Path,
    model_name: str,
) -> list[tuple[float, float]]:
    """Return one (true_latency, median_predicted_latency) pair per kernel.

    Each kernel directory holds repeated LLM estimate samples for the same
    fixed design, so the true latency is constant across samples.
    """
    kernel_medians = []

    for results_path in sorted(output_data_dir.glob("*/all_eval_data.json")):
        all_results = json.loads(results_path.read_text())
        model_results = [
            sample
            for sample in all_results.values()
            if sample.get("model_name") == model_name
        ]
        if not model_results:
            continue

        true_latency = model_results[0].get("target_actual_latency_cycles__cosim")
        estimated_latencies = [
            sample["estimated_latency_cycles"]
            for sample in model_results
            if sample.get("estimated_latency_cycles") is not None
        ]

        if true_latency is None or not estimated_latencies:
            continue

        kernel_medians.append((true_latency, statistics.median(estimated_latencies)))

    return kernel_medians


def plot_true_vs_predicted(
    true_vs_predicted: list[tuple[float, float]],
    plot_path: Path,
    model_name: str,
    kernel_median_true_vs_predicted: list[tuple[float, float]] | None = None,
) -> None:
    if not true_vs_predicted:
        raise ValueError("No valid model results were found")

    true_values = [true for true, _ in true_vs_predicted]
    predicted_values = [predicted for _, predicted in true_vs_predicted]

    figure, axis = plt.subplots(figsize=(5.5, 5.5))

    axis.scatter(
        true_values,
        predicted_values,
        s=36,
        facecolor=(0.122, 0.467, 0.706, 0.5),
        edgecolor="tab:blue",
        linewidth=0.5,
        zorder=2,
    )

    all_values = true_values + predicted_values
    if kernel_median_true_vs_predicted:
        median_true_values = [true for true, _ in kernel_median_true_vs_predicted]
        median_predicted_values = [
            predicted for _, predicted in kernel_median_true_vs_predicted
        ]
        all_values += median_true_values + median_predicted_values

        axis.scatter(
            median_true_values,
            median_predicted_values,
            s=22,
            marker="x",
            color=(0.8, 0.0, 0.0, 0.4),
            linewidth=1.2,
            label="Per-kernel median",
            zorder=3,
        )
        axis.legend(loc="upper left", bbox_to_anchor=(0.03, 0.80), fontsize=8, framealpha=0.85)

    axis_min = min(all_values) * 0.8
    axis_max = max(all_values) * 1.2
    axis.plot(
        [axis_min, axis_max],
        [axis_min, axis_max],
        linestyle="--",
        color="red",
        linewidth=1.2,
    )

    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(axis_min, axis_max)
    axis.set_ylim(axis_min, axis_max)
    axis.set_aspect("equal", adjustable="box")

    box_style = dict(
        boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.85
    )
    axis.text(
        0.03,
        0.94,
        "Above line:\npredicted slower than true",
        transform=axis.transAxes,
        fontsize=8,
        ha="left",
        va="top",
        bbox=box_style,
    )
    axis.text(
        0.97,
        0.06,
        "Below line:\npredicted faster than true",
        transform=axis.transAxes,
        fontsize=8,
        ha="right",
        va="bottom",
        bbox=box_style,
    )

    axis.set_title(f"HLS Kernel Latency: Predicted vs. True - {model_name}")
    axis.set_xlabel("True Latency (Clock Cycles, Log Scale)")
    axis.set_ylabel("Predicted Latency (Clock Cycles, Log Scale)")
    axis.grid(linestyle="--", alpha=0.35)
    axis.set_axisbelow(True)

    figure.tight_layout()
    figure.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_residuals(
    true_vs_predicted: list[tuple[float, float]],
    plot_path: Path,
    model_name: str,
    kernel_median_true_vs_predicted: list[tuple[float, float]] | None = None,
) -> None:
    if not true_vs_predicted:
        raise ValueError("No valid model results were found")

    true_values = [true for true, _ in true_vs_predicted]
    signed_errors = [predicted - true for true, predicted in true_vs_predicted]

    figure, axis = plt.subplots(figsize=(7, 5))

    axis.axhline(0.0, color="red", linestyle="--", linewidth=1.2, zorder=1)
    axis.scatter(
        true_values,
        signed_errors,
        s=36,
        facecolor=(0.122, 0.467, 0.706, 0.5),
        edgecolor="tab:blue",
        linewidth=0.5,
        zorder=2,
    )

    all_signed_errors = list(signed_errors)
    if kernel_median_true_vs_predicted:
        median_true_values = [true for true, _ in kernel_median_true_vs_predicted]
        median_signed_errors = [
            predicted - true for true, predicted in kernel_median_true_vs_predicted
        ]
        all_signed_errors += median_signed_errors

        axis.scatter(
            median_true_values,
            median_signed_errors,
            s=22,
            marker="x",
            color=(0.8, 0.0, 0.0, 0.4),
            linewidth=1.2,
            label="Per-kernel median",
            zorder=3,
        )
        axis.legend(loc="lower left", fontsize=8, framealpha=0.85)

    axis.set_xscale("log")
    linear_threshold = max(
        1.0, min(abs(e) for e in all_signed_errors if e != 0) or 1.0
    )
    axis.set_yscale("symlog", linthresh=linear_threshold)

    y_min = min(all_signed_errors)
    y_max = max(all_signed_errors)
    y_buffer = max(abs(y_min), abs(y_max), linear_threshold) * 0.2
    axis.set_ylim(y_min - y_buffer, y_max + y_buffer)

    axis.set_title(f"HLS Kernel Latency Residuals - {model_name}")
    axis.set_xlabel("True Latency (Clock Cycles, Log Scale)")
    axis.set_ylabel("Signed Error: Predicted − True (Clock Cycles, Symlog Scale)")
    axis.grid(linestyle="--", alpha=0.35)
    axis.set_axisbelow(True)

    figure.tight_layout()
    figure.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_absolute_error_vs_true(
    true_vs_predicted: list[tuple[float, float]],
    plot_path: Path,
    model_name: str,
    kernel_median_true_vs_predicted: list[tuple[float, float]] | None = None,
) -> None:
    if not true_vs_predicted:
        raise ValueError("No valid model results were found")

    true_values = [true for true, _ in true_vs_predicted]
    absolute_errors = [abs(predicted - true) for true, predicted in true_vs_predicted]

    figure, axis = plt.subplots(figsize=(7, 5))

    axis.scatter(
        true_values,
        absolute_errors,
        s=36,
        facecolor=(0.122, 0.467, 0.706, 0.5),
        edgecolor="tab:blue",
        linewidth=0.5,
        zorder=2,
    )

    all_absolute_errors = list(absolute_errors)
    if kernel_median_true_vs_predicted:
        median_true_values = [true for true, _ in kernel_median_true_vs_predicted]
        median_absolute_errors = [
            abs(predicted - true) for true, predicted in kernel_median_true_vs_predicted
        ]
        all_absolute_errors += median_absolute_errors

        axis.scatter(
            median_true_values,
            median_absolute_errors,
            s=22,
            marker="x",
            color=(0.8, 0.0, 0.0, 0.4),
            linewidth=1.2,
            label="Per-kernel median",
            zorder=3,
        )
        axis.legend(loc="upper left", fontsize=8, framealpha=0.85)

    axis.set_xscale("log")
    axis.set_yscale("log")

    axis.set_title(f"HLS Kernel Latency Absolute Error vs. True Latency - {model_name}")
    axis.set_xlabel("True Latency (Clock Cycles, Log Scale)")
    axis.set_ylabel("Absolute Error: |Predicted − True| (Clock Cycles, Log Scale)")
    axis.grid(linestyle="--", alpha=0.35)
    axis.set_axisbelow(True)

    figure.tight_layout()
    figure.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def load_absolute_errors(
    output_data_dir: Path,
    model_name: str,
) -> list[float]:
    absolute_errors = []

    for results_path in sorted(output_data_dir.glob("*/all_eval_data.json")):
        all_results = json.loads(results_path.read_text())
        model_results = [
            sample
            for sample in all_results.values()
            if sample.get("model_name") == model_name
        ]

        for sample in model_results:
            estimated_cycles = sample.get("estimated_latency_cycles")
            actual_cycles = sample.get("target_actual_latency_cycles__cosim")

            if estimated_cycles is None or actual_cycles is None:
                continue

            absolute_errors.append(abs(estimated_cycles - actual_cycles))

    return absolute_errors


def plot_mae_histogram(
    absolute_errors: list[float],
    plot_path: Path,
    model_name: str,
) -> None:
    if not absolute_errors:
        raise ValueError("No valid model results were found")

    figure, axis = plt.subplots(figsize=(8, 5))

    sns.kdeplot(
        x=absolute_errors,
        ax=axis,
        log_scale=True,
        fill=True,
        color="tab:blue",
        linewidth=1.5,
    )

    axis.set_title(f"HLS Kernel Latency Absolute Error Distribution - {model_name}")
    axis.set_xlabel("Absolute Error (Clock Cycles, Log Scale)")
    axis.set_ylabel("Density")
    axis.grid(axis="y", linestyle="--", alpha=0.35)
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
        description="Plot predicted vs. true HLS kernel latency per model."
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
        help="Path for the generated predicted-vs-true PNG plot",
    )
    parser.add_argument(
        "--mae-hist-output",
        type=Path,
        default=DEFAULT_MAE_HIST_PLOT_PATH,
        help="Path for the generated MAE histogram PNG plot",
    )
    parser.add_argument(
        "--residual-output",
        type=Path,
        default=DEFAULT_RESIDUAL_PLOT_PATH,
        help="Path for the generated residual (signed error vs. true latency) PNG plot",
    )
    parser.add_argument(
        "--abs-error-vs-true-output",
        type=Path,
        default=DEFAULT_ABS_ERROR_VS_TRUE_PLOT_PATH,
        help="Path for the generated absolute error vs. true latency PNG plot",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.mae_hist_output.parent.mkdir(parents=True, exist_ok=True)
    args.residual_output.parent.mkdir(parents=True, exist_ok=True)
    args.abs_error_vs_true_output.parent.mkdir(parents=True, exist_ok=True)

    for model_name in MODELS_TO_PLOT:
        plot_path = model_plot_path(args.output, model_name)
        true_vs_predicted = load_true_vs_predicted(args.output_data_dir, model_name)
        if true_vs_predicted:
            plot_true_vs_predicted(true_vs_predicted, plot_path, model_name)
            print(f"Saved {model_name} plot to {plot_path}")

            residual_plot_path = model_plot_path(args.residual_output, model_name)
            plot_residuals(true_vs_predicted, residual_plot_path, model_name)
            print(f"Saved {model_name} residual plot to {residual_plot_path}")

            abs_error_vs_true_plot_path = model_plot_path(
                args.abs_error_vs_true_output, model_name
            )
            plot_absolute_error_vs_true(
                true_vs_predicted, abs_error_vs_true_plot_path, model_name
            )
            print(
                f"Saved {model_name} absolute error vs. true latency plot to "
                f"{abs_error_vs_true_plot_path}"
            )
        else:
            print(f"Skipping {model_name} plot: no valid model results")

        mae_hist_plot_path = model_plot_path(args.mae_hist_output, model_name)
        absolute_errors = load_absolute_errors(args.output_data_dir, model_name)
        if absolute_errors:
            plot_mae_histogram(absolute_errors, mae_hist_plot_path, model_name)
            print(f"Saved {model_name} MAE histogram to {mae_hist_plot_path}")
        else:
            print(f"Skipping {model_name} MAE histogram: no valid model results")


if __name__ == "__main__":
    main()
