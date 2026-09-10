import argparse
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import textalloc as ta
from scipy.stats import kendalltau, spearmanr

DIR_CURRENT = Path(__file__).resolve().parent
DIR_FIGURES = DIR_CURRENT / "figures"

DEFAULT_OUTPUT_DATA_DIR = DIR_CURRENT / "output_data"
DEFAULT_RANK_SCATTER_PLOT_PATH = DIR_FIGURES / "kernel_latency_rank_scatter.png"
DEFAULT_SLOPE_PLOT_PATH = DIR_FIGURES / "kernel_latency_rank_slope.png"

MODELS_TO_PLOT = ["deepseek/deepseek-v4-flash", "openai/gpt-oss-120b"]


AGGREGATORS = {
    "median": statistics.median,
    "mean": statistics.mean,
}


def load_kernel_rankings(
    output_data_dir: Path,
    model_name: str,
    aggregator: str = "median",
) -> list[tuple[str, float, float]]:
    """Return (kernel_name, true_latency, predicted_latency) for each kernel.

    Each kernel directory holds repeated LLM estimate samples for the same
    fixed design, so the true latency is constant across samples and the
    predicted latency is the chosen aggregate (median by default, since a
    single wildly-off sample can otherwise skew the mean) of the valid
    (non-null) estimates.
    """
    aggregate = AGGREGATORS[aggregator]
    kernel_latencies = []

    for results_path in sorted(output_data_dir.glob("*/all_eval_data.json")):
        all_results = json.loads(results_path.read_text())
        model_results = [
            sample
            for sample in all_results.values()
            if sample.get("model_name") == model_name
        ]
        if not model_results:
            continue

        kernel_name = model_results[0]["benchmark_case_name"]
        true_latency = model_results[0].get("target_actual_latency_cycles__cosim")
        estimated_latencies = [
            sample["estimated_latency_cycles"]
            for sample in model_results
            if sample.get("estimated_latency_cycles") is not None
        ]

        if true_latency is None or not estimated_latencies:
            continue

        predicted_latency = aggregate(estimated_latencies)
        kernel_latencies.append((kernel_name, true_latency, predicted_latency))

    return kernel_latencies


def compute_rankings(
    kernel_latencies: list[tuple[str, float, float]],
) -> tuple[list[str], list[int], list[int], float, float]:
    """Rank kernels slowest-to-fastest by true and predicted latency.

    Returns (kernel_names, true_ranks, predicted_ranks, spearman_rho, kendall_tau),
    where rank 1 is the slowest (highest latency) design.
    """
    kernel_names = [name for name, _, _ in kernel_latencies]
    true_values = [true for _, true, _ in kernel_latencies]
    predicted_values = [predicted for _, _, predicted in kernel_latencies]

    def to_ranks(values: list[float]) -> list[int]:
        # Rank 1 = slowest (largest latency).
        order = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
        ranks = [0] * len(values)
        for rank, index in enumerate(order, start=1):
            ranks[index] = rank
        return ranks

    true_ranks = to_ranks(true_values)
    predicted_ranks = to_ranks(predicted_values)

    spearman_rho, _ = spearmanr(true_ranks, predicted_ranks)
    kendall_tau, _ = kendalltau(true_ranks, predicted_ranks)

    return kernel_names, true_ranks, predicted_ranks, spearman_rho, kendall_tau


def plot_rank_scatter(
    kernel_names: list[str],
    true_ranks: list[int],
    predicted_ranks: list[int],
    spearman_rho: float,
    kendall_tau: float,
    plot_path: Path,
    model_name: str,
) -> None:
    n_kernels = len(kernel_names)
    figure, axis = plt.subplots(figsize=(6, 6))

    axis.scatter(
        true_ranks,
        predicted_ranks,
        s=42,
        facecolor=(0.122, 0.467, 0.706, 0.55),
        edgecolor="tab:blue",
        linewidth=0.6,
        zorder=2,
    )

    axis.set_xlim(0.5, n_kernels + 1.5)
    axis.set_ylim(0.5, n_kernels + 1.5)
    axis.set_aspect("equal", adjustable="box")

    ta.allocate(
        axis,
        true_ranks,
        predicted_ranks,
        kernel_names,
        x_scatter=true_ranks,
        y_scatter=predicted_ranks,
        textsize=6,
        textcolor="dimgray",
        linewidth=0.5,
        linecolor="gray",
        min_distance=0.02,
        max_distance=0.5,
        nbr_candidates=400,
        draw_all=True,
    )

    x_min, x_max = axis.get_xlim()
    y_min, y_max = axis.get_ylim()
    diagonal_min = min(x_min, y_min)
    diagonal_max = max(x_max, y_max)
    axis.plot(
        [diagonal_min, diagonal_max],
        [diagonal_min, diagonal_max],
        linestyle="--",
        color="red",
        linewidth=1.2,
        zorder=1,
    )
    axis.set_xlim(x_min, x_max)
    axis.set_ylim(y_min, y_max)

    axis.set_title(
        f"HLS Kernel Latency Ranking: Predicted vs. True - {model_name}\n"
        f"Spearman ρ = {spearman_rho:.3f}, Kendall τ = {kendall_tau:.3f}"
    )
    axis.set_xlabel("True Rank (1 = slowest)")
    axis.set_ylabel("Predicted Rank (1 = slowest)")
    axis.grid(linestyle="--", alpha=0.35)
    axis.set_axisbelow(True)

    figure.tight_layout()
    figure.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_rank_slope(
    kernel_names: list[str],
    true_ranks: list[int],
    predicted_ranks: list[int],
    spearman_rho: float,
    kendall_tau: float,
    plot_path: Path,
    model_name: str,
) -> None:
    n_kernels = len(kernel_names)
    order = sorted(range(n_kernels), key=lambda i: true_ranks[i])

    figure_height = max(6.0, 0.32 * n_kernels)
    figure, axis = plt.subplots(figsize=(6.5, figure_height))

    for index in order:
        true_rank = true_ranks[index]
        predicted_rank = predicted_ranks[index]
        rank_error = abs(true_rank - predicted_rank)
        color = "tab:blue" if rank_error <= max(2, n_kernels // 8) else "tab:red"
        axis.plot(
            [0, 1],
            [true_rank, predicted_rank],
            color=color,
            alpha=0.6,
            linewidth=1.3,
            marker="o",
            markersize=4,
        )

    axis.set_xlim(-0.15, 1.15)
    axis.set_xticks([0, 1])
    axis.set_xticklabels(["True Rank", "Predicted Rank"])

    axis.set_ylim(n_kernels + 0.6, 0.4)
    axis.set_yticks(range(1, n_kernels + 1))
    true_order_labels = [
        kernel_names[i] for i in sorted(order, key=lambda i: true_ranks[i])
    ]
    axis.set_yticklabels(true_order_labels, fontsize=7.5)

    axis.set_title(
        f"HLS Kernel Latency Ranking (Slowest → Fastest) - {model_name}\n"
        f"Spearman ρ = {spearman_rho:.3f}, Kendall τ = {kendall_tau:.3f}"
    )
    axis.grid(axis="y", linestyle="--", alpha=0.25)
    axis.set_axisbelow(True)

    figure.tight_layout()
    figure.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def model_plot_path(
    base_path: Path, model_name: str, aggregator: str | None = None
) -> Path:
    normalized_model_name = (
        model_name.replace("/", "_").replace("-", "_").replace(" ", "_").lower()
    )
    suffix = f"__{normalized_model_name}"
    if aggregator is not None:
        suffix += f"__{aggregator}"
    return base_path.with_name(f"{base_path.stem}{suffix}{base_path.suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score and plot how well each model ranks HLS kernel designs "
        "by latency, relative to the gold (true) ranking."
    )
    parser.add_argument(
        "--output-data-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DATA_DIR,
        help="Directory containing per-kernel evaluation result directories",
    )
    parser.add_argument(
        "--rank-scatter-output",
        type=Path,
        default=DEFAULT_RANK_SCATTER_PLOT_PATH,
        help="Path for the generated rank scatter PNG plot",
    )
    parser.add_argument(
        "--rank-slope-output",
        type=Path,
        default=DEFAULT_SLOPE_PLOT_PATH,
        help="Path for the generated rank slope PNG plot",
    )
    parser.add_argument(
        "--aggregator",
        choices=sorted(AGGREGATORS),
        default="median",
        help="How to combine repeated LLM estimate samples per kernel",
    )
    args = parser.parse_args()

    args.rank_scatter_output.parent.mkdir(parents=True, exist_ok=True)
    args.rank_slope_output.parent.mkdir(parents=True, exist_ok=True)

    for model_name in MODELS_TO_PLOT:
        kernel_latencies = load_kernel_rankings(
            args.output_data_dir, model_name, args.aggregator
        )
        if not kernel_latencies:
            print(f"Skipping {model_name} ranking: no valid model results")
            continue

        kernel_names, true_ranks, predicted_ranks, spearman_rho, kendall_tau = (
            compute_rankings(kernel_latencies)
        )
        print(
            f"{model_name}: n={len(kernel_names)} "
            f"Spearman rho={spearman_rho:.3f} Kendall tau={kendall_tau:.3f}"
        )

        plot_model_label = f"{model_name} ({args.aggregator})"

        scatter_path = model_plot_path(
            args.rank_scatter_output, model_name, args.aggregator
        )
        plot_rank_scatter(
            kernel_names,
            true_ranks,
            predicted_ranks,
            spearman_rho,
            kendall_tau,
            scatter_path,
            plot_model_label,
        )
        print(f"Saved {model_name} rank scatter plot to {scatter_path}")

        slope_path = model_plot_path(
            args.rank_slope_output, model_name, args.aggregator
        )
        plot_rank_slope(
            kernel_names,
            true_ranks,
            predicted_ranks,
            spearman_rho,
            kendall_tau,
            slope_path,
            plot_model_label,
        )
        print(f"Saved {model_name} rank slope plot to {slope_path}")


if __name__ == "__main__":
    main()
