import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_weights(npz_path: Path):
    d = np.load(npz_path, allow_pickle=True)

    # Common keys used in this repo.
    for key in ("weights", "w_in", "W", "w"):
        if key in d:
            w = d[key].astype(np.float32)
            return w, key

    # Fallback: first 2D array in archive.
    for key in d.files:
        arr = d[key]
        if isinstance(arr, np.ndarray) and arr.ndim == 2:
            return arr.astype(np.float32), key

    raise ValueError(f"No 2D weight array found in {npz_path}")


def summarize_changes(w_before, w_after, eps=1e-8):
    if w_before.shape != w_after.shape:
        raise ValueError(f"Shape mismatch: before={w_before.shape}, after={w_after.shape}")

    diff = w_after - w_before
    abs_diff = np.abs(diff)

    num_total = diff.size
    changed_mask = abs_diff > eps
    num_changed = int(np.count_nonzero(changed_mask))
    pct_changed = 100.0 * num_changed / max(1, num_total)

    stats = {
        "shape": list(diff.shape),
        "num_total_weights": int(num_total),
        "num_changed_weights": num_changed,
        "percent_changed": pct_changed,
        "mean_abs_change": float(abs_diff.mean()),
        "max_abs_change": float(abs_diff.max()),
        "mean_signed_change": float(diff.mean()),
        "l1_change": float(abs_diff.sum()),
        "l2_change": float(np.sqrt(np.sum(diff * diff))),
        "num_increased": int(np.count_nonzero(diff > eps)),
        "num_decreased": int(np.count_nonzero(diff < -eps)),
        "num_unchanged": int(num_total - num_changed),
    }

    # Per-neuron (column) analysis.
    col_mean_abs = abs_diff.mean(axis=0)
    col_idx_max = int(np.argmax(col_mean_abs))
    stats["most_changed_column"] = col_idx_max
    stats["most_changed_column_mean_abs"] = float(col_mean_abs[col_idx_max])

    return stats, diff, abs_diff, col_mean_abs


def save_plot(abs_diff, col_mean_abs, out_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    ax0, ax1, ax2 = axes

    # 1) Heatmap of absolute changes.
    im = ax0.imshow(abs_diff, aspect="auto", cmap="magma")
    ax0.set_title("Absolute Weight Change Heatmap")
    ax0.set_xlabel("Post neuron index")
    ax0.set_ylabel("Pre neuron index")
    fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)

    # 2) Histogram of absolute changes.
    ax1.hist(abs_diff.ravel(), bins=60, color="steelblue", edgecolor="black", alpha=0.8)
    ax1.set_title("Distribution of |Δw|")
    ax1.set_xlabel("|Δw|")
    ax1.set_ylabel("Count")

    # 3) Mean |Δw| per output neuron.
    ax2.plot(np.arange(len(col_mean_abs)), col_mean_abs, color="darkorange")
    ax2.set_title("Mean |Δw| per Output Neuron")
    ax2.set_xlabel("Post neuron index")
    ax2.set_ylabel("Mean |Δw|")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"Saved plot: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare weights before vs after training.")
    parser.add_argument("--before", type=Path, required=True, help="Path to pre-training .npz weight/model file")
    parser.add_argument("--after", type=Path, required=True, help="Path to post-training .npz weight/model file")
    parser.add_argument("--eps", type=float, default=1e-8, help="Threshold to count a weight as changed")
    parser.add_argument(
        "--report-out",
        type=Path,
        default=Path("reports/weight_change_report.json"),
        help="Where to save JSON summary",
    )
    parser.add_argument(
        "--plot-out",
        type=Path,
        default=None,
        help="Optional output path for visualization image (e.g., reports/weight_change.png)",
    )
    args = parser.parse_args()

    w_before, key_before = load_weights(args.before)
    w_after, key_after = load_weights(args.after)

    stats, _diff, abs_diff, col_mean_abs = summarize_changes(w_before, w_after, eps=args.eps)
    stats["before_file"] = str(args.before)
    stats["after_file"] = str(args.after)
    stats["before_key"] = key_before
    stats["after_key"] = key_after

    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"Saved report: {args.report_out}")

    print("=== Weight Change Summary ===")
    print(f"Shape: {tuple(stats['shape'])}")
    print(f"Changed: {stats['num_changed_weights']}/{stats['num_total_weights']} ({stats['percent_changed']:.4f}%)")
    print(f"Mean |dw|: {stats['mean_abs_change']:.8f}")
    print(f"Max  |dw|: {stats['max_abs_change']:.8f}")
    print(f"L1 change: {stats['l1_change']:.8f}")
    print(f"L2 change: {stats['l2_change']:.8f}")
    print(f"Increased: {stats['num_increased']}, Decreased: {stats['num_decreased']}, Unchanged: {stats['num_unchanged']}")
    print(
        f"Most changed column: {stats['most_changed_column']} "
        f"(mean |dw|={stats['most_changed_column_mean_abs']:.8f})"
    )

    if args.plot_out is not None:
        save_plot(abs_diff, col_mean_abs, args.plot_out)


if __name__ == "__main__":
    main()
