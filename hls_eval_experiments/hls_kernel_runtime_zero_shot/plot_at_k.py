import argparse
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, LogLocator, NullFormatter, ScalarFormatter
from scipy.stats import rankdata

DIR_CURRENT = Path(__file__).resolve().parent
DIR_FIGURES = DIR_CURRENT / "figures"

DEFAULT_OUTPUT_DATA_DIR = DIR_CURRENT / "output_data"
DEFAULT_TAU_AT_K_PLOT_PATH = DIR_FIGURES / "kernel_latency_tau_at_k.png"
DEFAULT_RHO_AT_K_PLOT_PATH = DIR_FIGURES / "kernel_latency_rho_at_k.png"

MODELS_TO_PLOT = ["deepseek/deepseek-v4-flash", "openai/gpt-oss-120b"]

# Monte Carlo trials per k. A pilot run measured the per-k standard deviation
# of rho/tau under resampling (largest at k=1, shrinking as k grows toward
# the max sample count) and solved trials = (sigma / target_se)^2 for a
# target standard error of 0.005; the largest requirement observed was ~153
# trials at k=1. This is cheap CPU-only resampling of already-collected LLM
# estimates (no new LLM calls), so we use a generous fixed trial count well
# above that minimum rather than tuning it per k.
MONTE_CARLO_TRIALS = 10_000

AGGREGATORS = {
    "median": statistics.median,
    "mean": statistics.mean,
}


def load_kernel_samples(
    output_data_dir: Path,
    model_name: str,
) -> list[tuple[str, float, list[float]]]:
    """Return (kernel_name, true_latency, valid_estimates) for each kernel.

    Kernels with zero valid (non-null) estimates are excluded entirely, since
    there is no sample to draw from at any k.
    """
    kernels = []

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

        kernels.append((kernel_name, true_latency, estimated_latencies))

    return kernels


def draw_k_samples_batch(
    estimates: list[float], k: int, n_trials: int, rng: np.random.Generator
) -> np.ndarray:
    """Draw k samples for one kernel, independently for each of n_trials trials.

    Returns an (n_trials, k) array. Sampled without replacement when the
    kernel has at least k valid estimates (an ordinary random subset, as in
    pass@k): each trial row is obtained by independently shuffling the full
    estimate array and keeping the first k entries. When a kernel has fewer
    than k valid estimates, without-replacement sampling of k values is
    impossible, so we fall back to sampling with replacement from whatever
    valid estimates that kernel does have.
    """
    estimates_arr = np.asarray(estimates, dtype=float)
    n = estimates_arr.shape[0]

    if n >= k:
        # Independently permute each trial row, then take the first k
        # columns: equivalent to without-replacement sampling per trial.
        row_permutations = np.argsort(rng.random((n_trials, n)), axis=1)
        selected_indices = row_permutations[:, :k]
    else:
        selected_indices = rng.integers(0, n, size=(n_trials, k))

    return estimates_arr[selected_indices]


def aggregate_batch(samples: np.ndarray, aggregator_name: str) -> np.ndarray:
    """Aggregate an (n_trials, k) sample array to (n_trials,) along axis=1."""
    if aggregator_name == "median":
        return np.median(samples, axis=1)
    if aggregator_name == "mean":
        return np.mean(samples, axis=1)
    raise ValueError(f"Unknown aggregator: {aggregator_name}")


def spearman_rho_batch(true_values: np.ndarray, predicted_batch: np.ndarray) -> np.ndarray:
    """Vectorized Spearman rho for each trial (row) in predicted_batch.

    true_values: (n_kernels,) fixed gold values, shared across all trials.
    predicted_batch: (n_trials, n_kernels) predicted values, one row/trial.
    Returns (n_trials,) rho values, matching scipy.stats.spearmanr (which
    ranks with average-rank tie handling, then takes the Pearson correlation
    of the ranks).
    """
    true_ranks = rankdata(true_values)
    predicted_ranks = rankdata(predicted_batch, axis=1)

    true_centered = true_ranks - true_ranks.mean()
    predicted_centered = predicted_ranks - predicted_ranks.mean(axis=1, keepdims=True)

    numerator = predicted_centered @ true_centered
    denominator = np.sqrt(
        (predicted_centered**2).sum(axis=1) * (true_centered**2).sum()
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denominator != 0, numerator / denominator, np.nan)


def kendall_tau_b_batch(true_values: np.ndarray, predicted_batch: np.ndarray) -> np.ndarray:
    """Vectorized Kendall tau-b for each trial (row) in predicted_batch.

    Matches scipy.stats.kendalltau's default variant="b": pairwise
    concordant/discordant counts normalized by the geometric mean of
    (total pairs minus tied pairs) on each side, which reduces to the
    ordinary tau-a formula when there are no ties.
    """
    n_trials, n = predicted_batch.shape

    true_diff = true_values[:, None] - true_values[None, :]
    true_sign = np.sign(true_diff)

    predicted_diff = predicted_batch[:, :, None] - predicted_batch[:, None, :]
    predicted_sign = np.sign(predicted_diff)

    # Only count each unordered pair once (upper triangle, i < j).
    triu_i, triu_j = np.triu_indices(n, k=1)
    true_sign_pairs = true_sign[triu_i, triu_j]
    predicted_sign_pairs = predicted_sign[:, triu_i, triu_j]

    concordant_minus_discordant = (predicted_sign_pairs * true_sign_pairs[None, :]).sum(
        axis=1
    )

    ties_true = (true_sign_pairs == 0).sum()
    ties_predicted = (predicted_sign_pairs == 0).sum(axis=1)

    total_pairs = n * (n - 1) / 2
    denominator = np.sqrt(
        (total_pairs - ties_true) * (total_pairs - ties_predicted)
    )

    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(
            denominator != 0, concordant_minus_discordant / denominator, np.nan
        )


def run_monte_carlo_trials(
    kernels: list[tuple[str, float, list[float]]],
    k: int,
    aggregator_name: str,
    n_trials: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Run n_trials joint resampling trials at a given k.

    Each trial independently draws k samples per kernel (see
    draw_k_samples_batch), aggregates them into one predicted latency per
    kernel, and computes Spearman rho and Kendall tau between the resulting
    predicted-latency ranking and the gold ranking. All trials are computed
    together as vectorized numpy operations instead of a per-trial Python
    loop calling scipy.stats.

    Returns (rho_trials, tau_trials): the per-trial statistic arrays, from
    which the mean (rho@k / tau@k) and standard error can be computed.
    """
    true_values = np.asarray([true_latency for _, true_latency, _ in kernels])

    predicted_batch = np.stack(
        [
            aggregate_batch(
                draw_k_samples_batch(estimates, k, n_trials, rng), aggregator_name
            )
            for _, _, estimates in kernels
        ],
        axis=1,
    )

    rho_trials = spearman_rho_batch(true_values, predicted_batch)
    tau_trials = kendall_tau_b_batch(true_values, predicted_batch)

    return rho_trials, tau_trials


def compute_stat_at_k(
    kernels: list[tuple[str, float, list[float]]],
    aggregator_name: str,
    n_trials: int,
    seed: int,
) -> dict[int, dict[str, tuple[float, float]]]:
    """Compute rho@k and tau@k (mean and standard error) for k=1..max_n.

    max_n is the largest number of valid samples held by any included
    kernel. Returns {k: {"rho": (mean, se), "tau": (mean, se)}}.
    """
    max_n = max(len(estimates) for _, _, estimates in kernels)
    rng = np.random.default_rng(seed)

    results = {}
    for k in range(1, max_n + 1):
        rho_trials, tau_trials = run_monte_carlo_trials(
            kernels, k, aggregator_name, n_trials, rng
        )
        rho_mean = float(np.nanmean(rho_trials))
        rho_se = float(np.nanstd(rho_trials) / (n_trials**0.5))
        tau_mean = float(np.nanmean(tau_trials))
        tau_se = float(np.nanstd(tau_trials) / (n_trials**0.5))
        results[k] = {
            "rho": (rho_mean, rho_se),
            "tau": (tau_mean, tau_se),
        }

    return results


MODEL_COLORS = {
    "deepseek/deepseek-v4-flash": "tab:blue",
    "openai/gpt-oss-120b": "tab:orange",
}


def plot_stat_at_k(
    stat_at_k_by_model: dict[str, dict[int, dict[str, tuple[float, float]]]],
    stat_key: str,
    stat_label: str,
    plot_path: Path,
    aggregator: str,
) -> None:
    if not stat_at_k_by_model:
        raise ValueError("No valid model results were found")

    figure, axis = plt.subplots(figsize=(7.5, 5.5))

    all_k_values: set[int] = set()
    for model_name, stat_at_k in stat_at_k_by_model.items():
        k_values = sorted(stat_at_k)
        all_k_values.update(k_values)
        means = [stat_at_k[k][stat_key][0] for k in k_values]
        standard_errors = [stat_at_k[k][stat_key][1] for k in k_values]
        color = MODEL_COLORS.get(model_name)

        lower_band = [mean - 3 * se for mean, se in zip(means, standard_errors)]
        upper_band = [mean + 3 * se for mean, se in zip(means, standard_errors)]
        axis.fill_between(k_values, lower_band, upper_band, color=color, alpha=0.2)

        axis.plot(
            k_values,
            means,
            color=color,
            linewidth=1.6,
            marker="o",
            markersize=5,
            label=model_name,
        )

    sorted_k_values = sorted(all_k_values)
    axis.set_xscale("log")
    axis.set_xlim(min(sorted_k_values), max(sorted_k_values))
    axis.xaxis.set_major_locator(FixedLocator(sorted_k_values))
    axis.xaxis.set_major_formatter(ScalarFormatter())
    axis.xaxis.set_minor_locator(LogLocator(base=10, subs="all"))
    axis.xaxis.set_minor_formatter(NullFormatter())
    axis.set_ylim(0.0, 1.0)
    figure.suptitle(
        f"HLS Kernel Ranking {stat_label}@k ({aggregator})", fontsize=13, y=0.99
    )
    axis.set_title(
        "Shaded band: ±3 SE of the Monte Carlo estimate of the mean\n"
        "(sampling precision only, not ranking uncertainty)",
        fontsize=8.5,
        color="dimgray",
    )
    axis.set_xlabel("k (samples aggregated per kernel)")
    axis.set_ylabel(f"Monte Carlo Estimate of Expected {stat_label} vs. Gold Ranking")
    axis.grid(linestyle="--", alpha=0.35)
    axis.set_axisbelow(True)
    axis.legend(loc="lower right", fontsize=8, framealpha=0.85)

    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def aggregator_plot_path(base_path: Path, aggregator: str) -> Path:
    return base_path.with_name(f"{base_path.stem}__{aggregator}{base_path.suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot expected Kendall tau@k and Spearman rho@k between "
        "the LLM-predicted and gold HLS kernel latency rankings, as a "
        "function of how many samples per kernel are aggregated."
    )
    parser.add_argument(
        "--output-data-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DATA_DIR,
        help="Directory containing per-kernel evaluation result directories",
    )
    parser.add_argument(
        "--tau-at-k-output",
        type=Path,
        default=DEFAULT_TAU_AT_K_PLOT_PATH,
        help="Path for the generated tau@k PNG plot",
    )
    parser.add_argument(
        "--rho-at-k-output",
        type=Path,
        default=DEFAULT_RHO_AT_K_PLOT_PATH,
        help="Path for the generated rho@k PNG plot",
    )
    parser.add_argument(
        "--aggregator",
        choices=sorted(AGGREGATORS),
        default="median",
        help="How to combine the k sampled estimates per kernel",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=MONTE_CARLO_TRIALS,
        help="Monte Carlo trials per k",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for Monte Carlo sampling",
    )
    args = parser.parse_args()

    args.tau_at_k_output.parent.mkdir(parents=True, exist_ok=True)
    args.rho_at_k_output.parent.mkdir(parents=True, exist_ok=True)

    stat_at_k_by_model = {}
    for model_name in MODELS_TO_PLOT:
        kernels = load_kernel_samples(args.output_data_dir, model_name)
        if not kernels:
            print(f"Skipping {model_name}: no valid model results")
            continue

        stat_at_k = compute_stat_at_k(kernels, args.aggregator, args.trials, args.seed)
        stat_at_k_by_model[model_name] = stat_at_k
        for k in sorted(stat_at_k):
            rho_mean, _ = stat_at_k[k]["rho"]
            tau_mean, _ = stat_at_k[k]["tau"]
            print(f"{model_name} k={k}: rho@k={rho_mean:.4f} tau@k={tau_mean:.4f}")

    if not stat_at_k_by_model:
        return

    tau_plot_path = aggregator_plot_path(args.tau_at_k_output, args.aggregator)
    plot_stat_at_k(
        stat_at_k_by_model, "tau", "Kendall τ", tau_plot_path, args.aggregator
    )
    print(f"Saved tau@k plot to {tau_plot_path}")

    rho_plot_path = aggregator_plot_path(args.rho_at_k_output, args.aggregator)
    plot_stat_at_k(
        stat_at_k_by_model, "rho", "Spearman ρ", rho_plot_path, args.aggregator
    )
    print(f"Saved rho@k plot to {rho_plot_path}")


if __name__ == "__main__":
    main()
