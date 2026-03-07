"""
generate_plots.py
=================
Generate all publication-quality plots from experiment results.

Reads CSV data from results/raw/ and produces figures in plots/:
  1. Accuracy vs SRAM Reads (tradeoff frontier)
  2. Accuracy vs Latency
  3. Histogram of exit times
  4. Histogram of reads saved
  5. Confidence vs timestep trajectory
  6. Ablation comparison bar chart

Usage:
    python generate_plots.py                          # Default paths
    python generate_plots.py --results-dir results    # Custom results dir
"""

import os
import glob
import argparse
import csv
import json
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib


def load_hw_profiles(results_dir):
    """Load all hardware profile CSVs from results/raw/."""
    raw_dir = os.path.join(results_dir, "raw")
    all_profiles = defaultdict(list)

    for fpath in sorted(glob.glob(os.path.join(raw_dir, "*_hw_profiles.csv"))):
        fname = os.path.basename(fpath)
        # Parse config name from filename: configname_seedN_hw_profiles.csv
        parts = fname.replace("_hw_profiles.csv", "").rsplit("_seed", 1)
        config_name = parts[0] if parts else fname

        with open(fpath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                row['config'] = config_name
                # Convert numeric fields
                for k in row:
                    if k in ('config',):
                        continue
                    try:
                        row[k] = float(row[k])
                    except (ValueError, TypeError):
                        pass
                all_profiles[config_name].append(row)

    return all_profiles


def load_summaries(results_dir):
    """Load all run summary JSONs."""
    raw_dir = os.path.join(results_dir, "raw")
    summaries = defaultdict(list)

    for fpath in sorted(glob.glob(os.path.join(raw_dir, "*_summary.json"))):
        with open(fpath) as f:
            data = json.load(f)
        config_name = data.get("experiment", "unknown")
        summaries[config_name].append(data)

    return summaries


def plot_accuracy_vs_sram_reads(summaries, outdir):
    """Plot 1: Accuracy vs SRAM Reads tradeoff frontier."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for config_name, runs in sorted(summaries.items()):
        accs = [r.get("accuracy", 0) for r in runs]
        reads = [r.get("avg_sram_reads", 0) for r in runs]
        ax.scatter(reads, accs, s=80, label=config_name, zorder=3)
        if len(runs) > 1:
            ax.errorbar(np.mean(reads), np.mean(accs),
                        xerr=np.std(reads), yerr=np.std(accs),
                        fmt='none', capsize=4, color='gray', alpha=0.5)

    ax.set_xlabel("Average SRAM Reads per Sample", fontsize=12)
    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.set_title("Accuracy vs Memory Reads Tradeoff", fontsize=14)
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "tradeoff_acc_vs_reads.png"), dpi=300)
    plt.close(fig)
    print(f"  Saved: tradeoff_acc_vs_reads.png")


def plot_accuracy_vs_latency(summaries, outdir):
    """Plot 2: Accuracy vs Latency (avg exit timestep)."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for config_name, runs in sorted(summaries.items()):
        accs = [r.get("accuracy", 0) for r in runs]
        latency = [r.get("avg_exit_timestep", 20) for r in runs]
        ax.scatter(latency, accs, s=80, label=config_name, zorder=3)

    ax.set_xlabel("Average Exit Timestep", fontsize=12)
    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.set_title("Accuracy vs Inference Latency", fontsize=14)
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "tradeoff_acc_vs_latency.png"), dpi=300)
    plt.close(fig)
    print(f"  Saved: tradeoff_acc_vs_latency.png")


def plot_exit_time_histogram(profiles, outdir):
    """Plot 3: Histogram of exit times per configuration."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for config_name, samples in sorted(profiles.items()):
        exit_times = [s.get("exit_timestep", 20) for s in samples]
        ax.hist(exit_times, bins=range(1, 22), alpha=0.6, label=config_name, edgecolor='black')

    ax.set_xlabel("Exit Timestep", fontsize=12)
    ax.set_ylabel("Number of Samples", fontsize=12)
    ax.set_title("Distribution of Early Exit Times", fontsize=14)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "histogram_exit_times.png"), dpi=300)
    plt.close(fig)
    print(f"  Saved: histogram_exit_times.png")


def plot_reads_saved_histogram(profiles, outdir):
    """Plot 4: Histogram of SRAM reads saved vs baseline."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Find baseline average reads: T_max=20 with no gatekeeper
    baseline_key = [k for k in profiles if "baseline" in k]
    if baseline_key:
        baseline_reads = np.mean([s.get("sram_reads_total", 0) for s in profiles[baseline_key[0]]])
    else:
        baseline_reads = None

    for config_name, samples in sorted(profiles.items()):
        if "baseline" in config_name:
            continue
        reads = [s.get("sram_reads_total", 0) for s in samples]
        if baseline_reads:
            saved = [baseline_reads - r for r in reads]
            ax.hist(saved, bins=30, alpha=0.6, label=config_name, edgecolor='black')

    if baseline_reads:
        ax.axvline(x=0, color='red', linestyle='--', label='Break-even')
        ax.set_xlabel("SRAM Reads Saved vs Baseline", fontsize=12)
    else:
        ax.set_xlabel("SRAM Reads", fontsize=12)

    ax.set_ylabel("Number of Samples", fontsize=12)
    ax.set_title("Distribution of Memory Reads Saved", fontsize=14)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "histogram_reads_saved.png"), dpi=300)
    plt.close(fig)
    print(f"  Saved: histogram_reads_saved.png")


def plot_ablation_bars(summaries, outdir):
    """Plot 5: Ablation comparison bar chart."""
    configs = sorted(summaries.keys())
    if not configs:
        return

    metrics = {
        "Accuracy (%)": "accuracy",
        "Avg Exit T": "avg_exit_timestep",
        "GK Rejection %": "gk_rejection_rate",
    }

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]

    for ax, (title, key) in zip(axes, metrics.items()):
        means, stds, labels = [], [], []
        for cfg in configs:
            values = [r.get(key, 0) for r in summaries[cfg]]
            if "rejection" in key.lower() or "rate" in key.lower():
                values = [v * 100 for v in values]
            means.append(np.mean(values))
            stds.append(np.std(values))
            labels.append(cfg.replace("ablation_", ""))

        x = np.arange(len(labels))
        ax.bar(x, means, yerr=stds, capsize=4, alpha=0.8, edgecolor='black')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.set_title(title, fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle("Ablation Study Results", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "ablation_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: ablation_comparison.png")


def main():
    parser = argparse.ArgumentParser(description="Generate publication plots")
    parser.add_argument('--results-dir', type=str, default='results',
                        help='Results directory')
    parser.add_argument('--plots-dir', type=str, default='plots',
                        help='Output plots directory')
    args = parser.parse_args()

    os.makedirs(args.plots_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Generating Publication Plots")
    print(f"{'='*60}\n")

    profiles = load_hw_profiles(args.results_dir)
    summaries = load_summaries(args.results_dir)

    if not summaries:
        print("No results found. Run experiments first:")
        print("  python run_ablation.py")
        return

    print(f"  Found {len(summaries)} configs, "
          f"{sum(len(v) for v in profiles.values())} sample profiles\n")

    plot_accuracy_vs_sram_reads(summaries, args.plots_dir)
    plot_accuracy_vs_latency(summaries, args.plots_dir)

    if profiles:
        plot_exit_time_histogram(profiles, args.plots_dir)
        plot_reads_saved_histogram(profiles, args.plots_dir)

    plot_ablation_bars(summaries, args.plots_dir)

    print(f"\n  All plots saved to {args.plots_dir}/")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
