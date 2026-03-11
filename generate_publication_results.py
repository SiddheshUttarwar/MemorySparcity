"""
generate_publication_results.py
================================
Generates ALL publication-grade results for the Sparse SNN paper.
Runs inference on the full test set, collects hardware metrics,
and produces 8 publication-quality plots + LaTeX-ready tables.

Usage (on Colab with GPU):
    !python generate_publication_results.py

Outputs:
    results/                         (directory)
    ├── per_sample_metrics.csv       (per-sample: exit_t, reads, savings, confidence)
    ├── summary_table.csv            (aggregated means ± std for paper table)
    ├── latex_table.tex              (copy-paste into your .tex file)
    ├── fig1_cumulative_sram.png     (baseline vs sparse SRAM reads over T)
    ├── fig2_exit_histogram.png      (early exit timestep distribution)
    ├── fig3_gatekeeper_bar.png      (gatekeeper filtering breakdown)
    ├── fig4_energy_comparison.png   (energy: SRAM + MAC for both models)
    ├── fig5_latency_speedup.png     (latency and throughput comparison)
    ├── fig6_firing_rate.png         (per-layer firing rates)
    ├── fig7_savings_scatter.png     (per-sample savings vs exit time)
    └── fig8_confidence_trajectory.png (confidence over time for sample digits)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import csv
import random
from collections import defaultdict

# ---- Config ----
NUM_SAMPLES = 500          # Use 500 for publication, 50 for quick test
SEED = 42
DPI = 300
T_MAX = 20

# Hardware energy model (45nm CMOS, from Horowitz 2014)
E_SRAM_PJ = 5.0           # pJ per 8-bit SRAM read (32KB)
E_MAC_PJ  = 0.2           # pJ per 8-bit integer MAC
CLK_NS    = 10.0           # ns per clock cycle (100 MHz)

# Ensure reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Plot style
matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': DPI,
    'savefig.dpi': DPI,
    'savefig.bbox': 'tight',
})

# Colors
C_BASE = '#E74C3C'   # red
C_SPARSE = '#00B4D8' # cyan
C_GK = '#FF6B9D'     # pink
C_ENERGY = '#7B61FF' # purple
C_EXIT = '#00E676'   # green
C_GOLD = '#FFD740'

def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  PUBLICATION RESULTS GENERATOR ({DEVICE})")
    print(f"  Samples: {NUM_SAMPLES} | Seed: {SEED}")
    print(f"{'='*60}\n")

    # ---- Create output dir ----
    os.makedirs('results', exist_ok=True)

    # ---- Import models ----
    from train import NMNISTDataset
    from snn_model import LeNet5_CSNN
    from sparse_snn_model import LeNet5_Sparse_CSNN

    # ---- Load models ----
    baseline = LeNet5_CSNN(in_channels=2, num_classes=10).to(DEVICE)
    sparse = LeNet5_Sparse_CSNN(in_channels=2, num_classes=10).to(DEVICE)

    try:
        baseline.load_state_dict(torch.load("best_baseline_model.pth", map_location=DEVICE))
        print("  ✅ Baseline model loaded")
    except FileNotFoundError:
        print("  ⚠️  Baseline model not found, using untrained")

    try:
        sparse.load_state_dict(torch.load("best_sparse_model.pth", map_location=DEVICE))
        print("  ✅ Sparse model loaded")
    except FileNotFoundError:
        print("  ❌ Sparse model not found — cannot continue")
        return

    baseline.eval()
    sparse.eval()
    baseline.sync_to_sram(); baseline.sync_from_sram()
    sparse.sync_to_sram(); sparse.sync_from_sram()

    # ---- Load test data ----
    test_ds = NMNISTDataset(data_dir='preprocessed_data_native', split='test')
    indices = random.sample(range(len(test_ds)), min(NUM_SAMPLES, len(test_ds)))

    # ---- Hooks for baseline internal spikes ----
    class SpikeHook:
        def __init__(self, mod):
            self.hook = mod.register_forward_hook(self._fn)
            self.spikes = None
        def _fn(self, mod, inp, out):
            dims = list(range(out.dim())); dims.remove(1)
            self.spikes = out.sum(dim=dims).detach().cpu().numpy()
        def close(self):
            self.hook.remove()

    hooks = [SpikeHook(m) for m in [baseline.lif1, baseline.lif2, baseline.lif3, baseline.lif4]]

    TOTAL_NEURONS = 32*28*28 + 64*14*14 + 128 + 10  # 37,770

    # ---- Storage ----
    results = []
    baseline_cumul_all = []
    sparse_cumul_all = []
    confidence_trajectories = {}  # digit -> list of trajectories
    per_layer_spikes_base = defaultdict(list)
    per_layer_spikes_sparse = defaultdict(list)

    baseline_correct = 0
    sparse_correct = 0

    print(f"\n  Running {NUM_SAMPLES} inferences...")

    for idx_i, data_idx in enumerate(indices):
        x_seq, label = test_ds[data_idx]
        x_seq = x_seq.unsqueeze(0).to(DEVICE)
        T = x_seq.size(1)

        # ---- Baseline inference ----
        with torch.no_grad():
            b_out = baseline(x_seq)

        # Baseline spike counting
        dims = list(range(x_seq.dim())); dims.remove(1)
        inp_spk_t = x_seq.sum(dim=dims).detach().cpu().numpy()
        int_spk_t = np.zeros(T)
        layer_names = ['lif1', 'lif2', 'lif3', 'lif4']
        for hi, h in enumerate(hooks):
            int_spk_t += h.spikes
            per_layer_spikes_base[layer_names[hi]].append(float(h.spikes.sum()))
        total_spk_t = inp_spk_t + int_spk_t
        b_cumul = np.cumsum(total_spk_t).tolist()
        b_reads = int(b_cumul[-1])

        b_pred = b_out.argmax(1).item() if b_out.dim() > 1 else b_out.argmax().item()
        if b_pred == label: baseline_correct += 1

        # ---- Sparse inference ----
        with torch.no_grad():
            spike_rate, _, l1_sum, actual_steps, hw = sparse(x_seq, early_exit=True)

        s_pred = spike_rate.argmax(1).item()
        if s_pred == label: sparse_correct += 1

        s_reads = hw['kept_in'] + int(l1_sum.item())
        s_cumul = list(hw['cumulative_reads_over_time'])
        while len(s_cumul) < T:
            s_cumul.append(s_cumul[-1])

        # Per-layer sparse spikes
        if 'per_layer_spikes' in hw:
            for ln, cnt in hw['per_layer_spikes'].items():
                per_layer_spikes_sparse[ln].append(float(cnt))

        saving_pct = 100 * (1 - s_reads / max(1, b_reads))
        gk_reject = 100 * (1 - hw['kept_in'] / max(1, hw['total_in']))

        b_firing = (int_spk_t.sum()) / (TOTAL_NEURONS * T) * 100
        s_firing = (l1_sum.item()) / (TOTAL_NEURONS * actual_steps) * 100

        # Energy
        e_base_sram = b_reads * E_SRAM_PJ
        e_base_mac = b_reads * E_MAC_PJ
        e_sparse_sram = s_reads * E_SRAM_PJ
        e_sparse_mac = s_reads * E_MAC_PJ
        e_base_total = e_base_sram + e_base_mac
        e_sparse_total = e_sparse_sram + e_sparse_mac

        # Latency
        lat_base = T * CLK_NS
        lat_sparse = actual_steps * CLK_NS

        # Confidence trajectory (save for a few representative digits)
        if 'confidence_trajectory' in hw and label not in confidence_trajectories:
            confidence_trajectories[label] = hw['confidence_trajectory']

        results.append({
            'sample': data_idx, 'label': label,
            'b_pred': b_pred, 's_pred': s_pred,
            'b_correct': int(b_pred == label), 's_correct': int(s_pred == label),
            'b_reads': b_reads, 's_reads': s_reads,
            'saving_pct': saving_pct,
            'exit_t': actual_steps,
            'gk_total': hw['total_in'], 'gk_kept': hw['kept_in'],
            'gk_reject_pct': gk_reject,
            'b_firing': b_firing, 's_firing': s_firing,
            'e_base': e_base_total, 'e_sparse': e_sparse_total,
            'lat_base': lat_base, 'lat_sparse': lat_sparse,
        })

        baseline_cumul_all.append(b_cumul)
        sparse_cumul_all.append(s_cumul)

        if (idx_i + 1) % 100 == 0:
            print(f"    [{idx_i+1}/{NUM_SAMPLES}] done")

    for h in hooks:
        h.close()

    N = len(results)
    print(f"\n  Inference complete. Baseline acc: {baseline_correct}/{N}, Sparse acc: {sparse_correct}/{N}")

    # ============================================================
    # EXPORT: Per-sample CSV
    # ============================================================
    csv_path = 'results/per_sample_metrics.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader(); w.writerows(results)
    print(f"  📄 {csv_path}")

    # ============================================================
    # AGGREGATE STATISTICS
    # ============================================================
    def agg(key):
        vals = [r[key] for r in results]
        return np.mean(vals), np.std(vals)

    stats = {
        'Accuracy (%)': (f"{baseline_correct/N*100:.1f}", f"{sparse_correct/N*100:.1f}"),
        'SRAM Reads': (f"{agg('b_reads')[0]:,.0f} ± {agg('b_reads')[1]:,.0f}",
                       f"{agg('s_reads')[0]:,.0f} ± {agg('s_reads')[1]:,.0f}"),
        'Reads Saved (%)': ('—', f"{agg('saving_pct')[0]:.1f} ± {agg('saving_pct')[1]:.1f}"),
        'GK Rejection (%)': ('0', f"{agg('gk_reject_pct')[0]:.1f} ± {agg('gk_reject_pct')[1]:.1f}"),
        'Exit Timestep': ('20', f"{agg('exit_t')[0]:.1f} ± {agg('exit_t')[1]:.1f}"),
        'Firing Rate (%)': (f"{agg('b_firing')[0]:.3f}", f"{agg('s_firing')[0]:.3f}"),
        'Energy (pJ)': (f"{agg('e_base')[0]:,.0f} ± {agg('e_base')[1]:,.0f}",
                        f"{agg('e_sparse')[0]:,.0f} ± {agg('e_sparse')[1]:,.0f}"),
        'Latency (ns)': (f"{agg('lat_base')[0]:.0f}", f"{agg('lat_sparse')[0]:.0f} ± {agg('lat_sparse')[1]:.0f}"),
    }

    # Summary CSV
    with open('results/summary_table.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Metric', 'Baseline', 'Sparse'])
        for k, (b, s) in stats.items():
            w.writerow([k, b, s])
    print(f"  📄 results/summary_table.csv")

    # LaTeX table
    with open('results/latex_table.tex', 'w') as f:
        f.write("\\begin{table}[t]\n\\caption{Baseline vs.\\ Sparse SNN — Hardware Metrics (N-MNIST, $N$=%d)}\n" % N)
        f.write("\\centering\\small\n\\begin{tabular}{lcc}\n\\hline\n")
        f.write("\\textbf{Metric} & \\textbf{Baseline} & \\textbf{Sparse} \\\\\n\\hline\n")
        for k, (b, s) in stats.items():
            k_esc = k.replace('%', '\\%').replace('_', '\\_')
            b_esc = b.replace('±', '$\\pm$')
            s_esc = s.replace('±', '$\\pm$')
            f.write(f"{k_esc} & {b_esc} & {s_esc} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\label{tab:hw_metrics}\n\\end{table}\n")
    print(f"  📄 results/latex_table.tex")

    # ============================================================
    # FIGURE 1: Cumulative SRAM Reads Over Time
    # ============================================================
    fig, ax = plt.subplots(figsize=(8, 5))
    t_axis = np.arange(1, T_MAX + 1)
    b_mean = np.mean(baseline_cumul_all, axis=0)
    b_std = np.std(baseline_cumul_all, axis=0)
    s_mean = np.mean(sparse_cumul_all, axis=0)
    s_std = np.std(sparse_cumul_all, axis=0)

    ax.fill_between(t_axis, b_mean - b_std, b_mean + b_std, alpha=0.15, color=C_BASE)
    ax.fill_between(t_axis, s_mean - s_std, s_mean + s_std, alpha=0.15, color=C_SPARSE)
    ax.plot(t_axis, b_mean, color=C_BASE, linewidth=2.5, marker='x', markersize=5, label='Baseline (Dense)')
    ax.plot(t_axis, s_mean, color=C_SPARSE, linewidth=2.5, marker='o', markersize=5, label='Sparse (Optimized)')

    # Shade savings area
    ax.fill_between(t_axis, s_mean, b_mean, alpha=0.08, color=C_GK, label='Reads Saved')

    mean_exit = agg('exit_t')[0]
    ax.axvline(x=mean_exit, color=C_EXIT, linestyle='--', linewidth=1.5, label=f'Avg Early Exit (T={mean_exit:.1f})')

    ax.set_xlabel('Timestep (T)')
    ax.set_ylabel('Cumulative SRAM Reads')
    ax.set_title(f'Cumulative SRAM Reads: Baseline vs. Sparse (N={N})')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(t_axis)
    fig.savefig('results/fig1_cumulative_sram.png')
    plt.close()
    print("  📊 fig1_cumulative_sram.png")

    # ============================================================
    # FIGURE 2: Early Exit Histogram
    # ============================================================
    exits = [r['exit_t'] for r in results]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.arange(0.5, T_MAX + 1.5, 1)
    ax.hist(exits, bins=bins, color=C_EXIT, edgecolor='#0a3520', alpha=0.85, rwidth=0.85)
    ax.axvline(x=np.mean(exits), color=C_BASE, linestyle='--', linewidth=2, label=f'Mean = {np.mean(exits):.1f}')
    ax.set_xlabel('Exit Timestep')
    ax.set_ylabel('Number of Samples')
    ax.set_title(f'Early Exit Timestep Distribution (N={N})')
    ax.set_xticks(range(1, T_MAX + 1))
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    fig.savefig('results/fig2_exit_histogram.png')
    plt.close()
    print("  📊 fig2_exit_histogram.png")

    # ============================================================
    # FIGURE 3: Gatekeeper Breakdown
    # ============================================================
    gk_kept = np.mean([r['gk_kept'] for r in results])
    gk_rejected = np.mean([r['gk_total'] - r['gk_kept'] for r in results])
    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(['Kept\n(Useful)', 'Rejected\n(Noise/Burst)'], [gk_kept, gk_rejected],
                  color=[C_SPARSE, C_GK], edgecolor='white', width=0.5)
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + max(gk_kept, gk_rejected)*0.02,
                f'{b.get_height():,.0f}', ha='center', fontweight='bold')
    ax.set_ylabel('Average Input Spikes per Inference')
    ax.set_title('Dynamic Gatekeeper: Input Spike Filtering')
    ax.grid(axis='y', alpha=0.3)
    fig.savefig('results/fig3_gatekeeper_bar.png')
    plt.close()
    print("  📊 fig3_gatekeeper_bar.png")

    # ============================================================
    # FIGURE 4: Energy Comparison
    # ============================================================
    e_b_sram = agg('e_base')[0] * E_SRAM_PJ / (E_SRAM_PJ + E_MAC_PJ)
    e_b_mac = agg('e_base')[0] * E_MAC_PJ / (E_SRAM_PJ + E_MAC_PJ)
    e_s_sram = agg('e_sparse')[0] * E_SRAM_PJ / (E_SRAM_PJ + E_MAC_PJ)
    e_s_mac = agg('e_sparse')[0] * E_MAC_PJ / (E_SRAM_PJ + E_MAC_PJ)

    fig, ax = plt.subplots(figsize=(6, 5))
    x = np.arange(2)
    w = 0.35
    b1 = ax.bar(x - w/2, [e_b_sram, e_s_sram], w, label='SRAM Energy', color=C_BASE, alpha=0.85)
    b2 = ax.bar(x + w/2, [e_b_mac, e_s_mac], w, label='MAC Energy', color=C_ENERGY, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(['Baseline', 'Sparse'])
    ax.set_ylabel('Energy (pJ)')
    ax.set_title('Energy Breakdown: SRAM vs. MAC')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Total energy labels
    for i, (total, label) in enumerate([(agg('e_base')[0], 'Baseline'), (agg('e_sparse')[0], 'Sparse')]):
        ax.text(i, max(e_b_sram, e_s_sram) * 1.05, f'Total: {total:,.0f} pJ', ha='center', fontweight='bold', fontsize=9)

    fig.savefig('results/fig4_energy_comparison.png')
    plt.close()
    print("  📊 fig4_energy_comparison.png")

    # ============================================================
    # FIGURE 5: Latency & Throughput
    # ============================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # Latency
    lat_b, lat_s = agg('lat_base')[0], agg('lat_sparse')[0]
    bars = ax1.bar(['Baseline', 'Sparse'], [lat_b, lat_s], color=[C_BASE, C_SPARSE], width=0.5)
    for b in bars:
        ax1.text(b.get_x() + b.get_width()/2, b.get_height() + lat_b * 0.02,
                 f'{b.get_height():.0f} ns', ha='center', fontweight='bold')
    ax1.set_ylabel('Latency per Inference (ns)')
    ax1.set_title('Inference Latency')
    ax1.grid(axis='y', alpha=0.3)

    # Throughput
    tp_b = 1e9 / lat_b / 1e6  # M inferences/sec
    tp_s = 1e9 / lat_s / 1e6
    bars2 = ax2.bar(['Baseline', 'Sparse'], [tp_b, tp_s], color=[C_BASE, C_SPARSE], width=0.5)
    for b in bars2:
        ax2.text(b.get_x() + b.get_width()/2, b.get_height() + tp_b * 0.02,
                 f'{b.get_height():.1f} M/s', ha='center', fontweight='bold')
    ax2.set_ylabel('Throughput (M inferences/sec)')
    ax2.set_title('Inference Throughput')
    ax2.grid(axis='y', alpha=0.3)

    fig.suptitle(f'Latency & Throughput Comparison @ 100 MHz', fontsize=13)
    plt.tight_layout()
    fig.savefig('results/fig5_latency_speedup.png')
    plt.close()
    print("  📊 fig5_latency_speedup.png")

    # ============================================================
    # FIGURE 6: Per-Layer Firing Rates
    # ============================================================
    layers = ['lif1', 'lif2', 'lif3', 'lif4']
    layer_labels = ['Conv1-LIF\n(32×28×28)', 'Conv2-LIF\n(64×14×14)', 'FC1-LIF\n(128)', 'FC2-LIF\n(10)']
    layer_neurons = [32*28*28, 64*14*14, 128, 10]

    b_rates = [np.mean(per_layer_spikes_base.get(l, [0])) / (n * T_MAX) * 100
               for l, n in zip(layers, layer_neurons)]
    # For sparse, use baseline layer names if sparse per-layer not available
    s_rates_raw = per_layer_spikes_sparse if per_layer_spikes_sparse else per_layer_spikes_base
    s_rates = []
    mean_exit_t = agg('exit_t')[0]
    for l, n in zip(layers, layer_neurons):
        vals = s_rates_raw.get(l, per_layer_spikes_base.get(l, [0]))
        s_rates.append(np.mean(vals) / (n * mean_exit_t) * 100)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(layers))
    w = 0.35
    ax.bar(x - w/2, b_rates, w, label='Baseline', color=C_BASE, alpha=0.85)
    ax.bar(x + w/2, s_rates, w, label='Sparse', color=C_SPARSE, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(layer_labels)
    ax.set_ylabel('Firing Rate (%)')
    ax.set_title('Per-Layer Average Firing Rate')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    fig.savefig('results/fig6_firing_rate.png')
    plt.close()
    print("  📊 fig6_firing_rate.png")

    # ============================================================
    # FIGURE 7: Per-Sample Savings Scatter
    # ============================================================
    fig, ax = plt.subplots(figsize=(7, 5))
    exits_arr = [r['exit_t'] for r in results]
    savings_arr = [r['saving_pct'] for r in results]
    scatter = ax.scatter(exits_arr, savings_arr, c=[r['label'] for r in results],
                         cmap='tab10', alpha=0.6, s=20, edgecolors='none')
    ax.set_xlabel('Early Exit Timestep')
    ax.set_ylabel('SRAM Reads Saved (%)')
    ax.set_title(f'Per-Sample Savings vs. Exit Time (N={N})')
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax, label='Digit Class')
    cbar.set_ticks(range(10))
    fig.savefig('results/fig7_savings_scatter.png')
    plt.close()
    print("  📊 fig7_savings_scatter.png")

    # ============================================================
    # FIGURE 8: Confidence Trajectory
    # ============================================================
    if confidence_trajectories:
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        for digit, traj in sorted(confidence_trajectories.items()):
            t_ax = range(1, len(traj) + 1)
            ax.plot(t_ax, traj, linewidth=2, color=colors[digit], label=f'Digit {digit}', marker='o', markersize=4)
        ax.axhline(y=0.9, color=C_GK, linestyle='--', linewidth=1.5, label='Exit Threshold (90%)')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Max Softmax Confidence')
        ax.set_title('Confidence Trajectory by Digit Class')
        ax.legend(ncol=3, fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(range(1, T_MAX + 1))
        fig.savefig('results/fig8_confidence_trajectory.png')
        plt.close()
        print("  📊 fig8_confidence_trajectory.png")

    # ============================================================
    # FINAL SUMMARY
    # ============================================================
    print(f"\n{'='*60}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Metric':<25} {'Baseline':>15} {'Sparse':>15}")
    print(f"  {'-'*55}")
    for k, (b, s) in stats.items():
        print(f"  {k:<25} {b:>15} {s:>15}")
    print(f"{'='*60}")
    print(f"\n  All outputs saved to results/")
    print(f"  Copy results/latex_table.tex directly into your paper!\n")


if __name__ == '__main__':
    main()
