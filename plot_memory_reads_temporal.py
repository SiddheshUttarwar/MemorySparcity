import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_model(model_path: Path):
    d = np.load(model_path, allow_pickle=True)
    if "w_in" not in d:
        raise ValueError(f"{model_path} does not contain 'w_in'")
    w_in = d["w_in"].astype(np.float32)
    gate_mask = d["gate_mask"].astype(np.uint8) if "gate_mask" in d else np.ones((w_in.shape[1],), dtype=np.uint8)
    return w_in, gate_mask


def load_sample(npz_path: Path):
    d = np.load(npz_path)
    if "x" not in d:
        raise ValueError(f"{npz_path} does not contain 'x'")
    x = d["x"].astype(np.float32)  # [T,2,H,W]
    y = int(d["y"]) if "y" in d else -1
    return x, y


def temporal_read_trace(x, hidden_dim, gate_mask=None):
    """
    Event-driven SRAM read estimate:
    reads_t = (#active presyn spikes at t) * (#target neurons read)

    Baseline: target neurons = hidden_dim
    Gated: target neurons = number of active neurons in gate_mask
    """
    t_steps = x.shape[0]
    x_flat = x.reshape(t_steps, -1)
    active_pre = (x_flat > 0).sum(axis=1).astype(np.int64)

    baseline_reads = active_pre * int(hidden_dim)
    if gate_mask is None:
        gated_reads = baseline_reads.copy()
        active_hidden = hidden_dim
    else:
        active_hidden = int(np.sum(gate_mask > 0))
        gated_reads = active_pre * active_hidden

    return active_pre, baseline_reads, gated_reads, active_hidden


def summarize_trace(reads):
    reads = reads.astype(np.float64)
    total = float(np.sum(reads))
    mean = float(np.mean(reads))
    peak = float(np.max(reads))
    std = float(np.std(reads))
    cv = std / mean if mean > 1e-12 else 0.0

    # Burst ratio: fraction of time bins above mean + 1*std
    thr = mean + std
    burst_ratio = float(np.mean(reads > thr))

    # Temporal concentration: top-20% time bins contribution
    if len(reads) == 0:
        top20_contrib = 0.0
    else:
        k = max(1, int(0.2 * len(reads)))
        top = np.sort(reads)[-k:]
        top20_contrib = float(np.sum(top) / max(total, 1e-12))

    return {
        "total_reads": total,
        "mean_reads_per_timestep": mean,
        "peak_reads_per_timestep": peak,
        "std_reads_per_timestep": std,
        "coef_variation": cv,
        "burst_ratio": burst_ratio,
        "top20_timebin_contribution": top20_contrib,
    }


def aggregate_samples(sample_paths, hidden_dim, gate_mask):
    all_active = []
    all_base = []
    all_gated = []
    per_sample = []

    for p in sample_paths:
        x, y = load_sample(p)
        active_pre, baseline_reads, gated_reads, active_hidden = temporal_read_trace(
            x, hidden_dim=hidden_dim, gate_mask=gate_mask
        )

        all_active.append(active_pre)
        all_base.append(baseline_reads)
        all_gated.append(gated_reads)

        base_s = summarize_trace(baseline_reads)
        gated_s = summarize_trace(gated_reads)
        reduction = (
            100.0 * (base_s["total_reads"] - gated_s["total_reads"]) / max(base_s["total_reads"], 1e-12)
        )
        per_sample.append(
            {
                "file": str(p),
                "label": y,
                "timesteps": int(len(baseline_reads)),
                "active_hidden_neurons": int(active_hidden),
                "baseline": base_s,
                "gated": gated_s,
                "read_reduction_percent": float(reduction),
            }
        )

    # Align by minimum length to compute mean temporal profile.
    min_t = min(len(a) for a in all_base)
    base_mat = np.stack([a[:min_t] for a in all_base], axis=0)
    gated_mat = np.stack([a[:min_t] for a in all_gated], axis=0)
    active_mat = np.stack([a[:min_t] for a in all_active], axis=0)

    mean_active = active_mat.mean(axis=0)
    mean_base = base_mat.mean(axis=0)
    mean_gated = gated_mat.mean(axis=0)
    mean_base_cum = np.cumsum(mean_base)
    mean_gated_cum = np.cumsum(mean_gated)

    overall_base = summarize_trace(mean_base)
    overall_gated = summarize_trace(mean_gated)
    overall_reduction = (
        100.0
        * (overall_base["total_reads"] - overall_gated["total_reads"])
        / max(overall_base["total_reads"], 1e-12)
    )

    aggregate = {
        "num_samples": len(sample_paths),
        "aligned_timesteps": int(min_t),
        "baseline_mean_trace": mean_base.tolist(),
        "gated_mean_trace": mean_gated.tolist(),
        "baseline_cumulative_mean_trace": mean_base_cum.tolist(),
        "gated_cumulative_mean_trace": mean_gated_cum.tolist(),
        "active_pre_mean_trace": mean_active.tolist(),
        "baseline_summary": overall_base,
        "gated_summary": overall_gated,
        "overall_read_reduction_percent": float(overall_reduction),
        "final_cumulative_baseline_reads": float(mean_base_cum[-1]) if len(mean_base_cum) else 0.0,
        "final_cumulative_gated_reads": float(mean_gated_cum[-1]) if len(mean_gated_cum) else 0.0,
    }
    return per_sample, aggregate, mean_active, mean_base, mean_gated, mean_base_cum, mean_gated_cum


def plot_temporal(mean_active, mean_base, mean_gated, mean_base_cum, mean_gated_cum, out_path: Path, title: str):
    t = np.arange(len(mean_base))
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax1 = axes[0]
    ax2 = ax1.twinx()

    ln1 = ax1.plot(t, mean_base, color="tab:blue", label="Baseline reads/t")
    ln2 = ax1.plot(t, mean_gated, color="tab:green", label="Gated reads/t")
    ln3 = ax2.plot(t, mean_active, color="tab:orange", alpha=0.7, label="Active pre spikes/t")

    ax1.set_xlabel("Time bin")
    ax1.set_ylabel("Memory reads")
    ax2.set_ylabel("Active pre spikes")
    ax1.set_title(title)
    ax1.grid(True, alpha=0.3)

    lines = ln1 + ln2 + ln3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right")

    ax3 = axes[1]
    ax3.plot(t, mean_base_cum, color="tab:blue", label="Baseline cumulative reads")
    ax3.plot(t, mean_gated_cum, color="tab:green", label="Gated cumulative reads")
    ax3.set_xlabel("Time bin")
    ax3.set_ylabel("Cumulative memory reads")
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="upper left")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"Saved temporal plot: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot temporal memory reads during testing and analyze read behavior.")
    parser.add_argument("--model-path", type=Path, required=True, help="Model .npz containing w_in and optional gate_mask")
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory of preprocessed .npz samples")
    parser.add_argument("--max-samples", type=int, default=10, help="Number of test samples to analyze")
    parser.add_argument("--plot-out", type=Path, default=Path("reports/temporal_memory_reads.png"))
    parser.add_argument("--report-out", type=Path, default=Path("reports/temporal_memory_reads_report.json"))
    args = parser.parse_args()

    w_in, gate_mask = load_model(args.model_path)
    hidden_dim = w_in.shape[1]

    sample_paths = sorted(args.data_dir.glob("*.npz"))
    if not sample_paths:
        raise FileNotFoundError(f"No .npz test samples found in {args.data_dir}")
    if args.max_samples > 0:
        sample_paths = sample_paths[: args.max_samples]

    per_sample, aggregate, mean_active, mean_base, mean_gated, mean_base_cum, mean_gated_cum = aggregate_samples(
        sample_paths, hidden_dim=hidden_dim, gate_mask=gate_mask
    )

    plot_temporal(
        mean_active,
        mean_base,
        mean_gated,
        mean_base_cum,
        mean_gated_cum,
        out_path=args.plot_out,
        title=f"Temporal Memory Reads ({args.model_path.name})",
    )

    report = {
        "model_path": str(args.model_path),
        "data_dir": str(args.data_dir),
        "hidden_dim": int(hidden_dim),
        "active_hidden_neurons_from_gate_mask": int(np.sum(gate_mask > 0)),
        "per_sample": per_sample,
        "aggregate": aggregate,
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved analysis report: {args.report_out}")

    base = aggregate["baseline_summary"]
    gated = aggregate["gated_summary"]
    print("=== Temporal Memory Read Analysis ===")
    print(f"Samples analyzed: {aggregate['num_samples']}")
    print(f"Timesteps aligned: {aggregate['aligned_timesteps']}")
    print(f"Baseline total reads (mean trace): {base['total_reads']:.2f}")
    print(f"Gated total reads (mean trace):    {gated['total_reads']:.2f}")
    print(f"Estimated read reduction:          {aggregate['overall_read_reduction_percent']:.2f}%")
    print(f"Cumulative baseline reads:         {aggregate['final_cumulative_baseline_reads']:.2f}")
    print(f"Cumulative gated reads:            {aggregate['final_cumulative_gated_reads']:.2f}")
    print(f"Burst ratio (baseline):            {base['burst_ratio']:.4f}")
    print(f"Top-20% timebin contribution:      {base['top20_timebin_contribution']:.4f}")


if __name__ == "__main__":
    main()
