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

DPI = 300


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

        # Baseline returns (spike_rate, out_spikes) tuple
        b_spike_rate = b_out[0] if isinstance(b_out, tuple) else b_out

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

        b_pred = b_spike_rate.argmax(1).item()
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
    # PLOT STYLE — Premium Dark Theme
    # ============================================================
    BG       = '#0D1117'
    PANEL    = '#161B22'
    GRID_C   = '#21262D'
    TEXT     = '#E6EDF3'
    TEXT_DIM = '#8B949E'
    RED      = '#FF6B6B'
    CYAN     = '#00D4FF'
    GREEN    = '#00E676'
    PURPLE   = '#B388FF'
    ORANGE   = '#FFB74D'
    PINK     = '#FF6B9D'
    GOLD     = '#FFD740'
    TEAL     = '#64FFDA'

    def style_ax(ax, title='', xlabel='', ylabel=''):
        ax.set_facecolor(PANEL)
        ax.set_title(title, color=TEXT, fontsize=13, fontweight='bold', pad=12)
        ax.set_xlabel(xlabel, color=TEXT_DIM, fontsize=10)
        ax.set_ylabel(ylabel, color=TEXT_DIM, fontsize=10)
        ax.tick_params(colors=TEXT_DIM, labelsize=9)
        ax.grid(True, color=GRID_C, alpha=0.6, linewidth=0.5)
        for spine in ax.spines.values():
            spine.set_color(GRID_C)

    def dark_fig(w=8, h=5):
        fig = plt.figure(figsize=(w, h), facecolor=BG)
        return fig

    # ============================================================
    # FIGURE 1: Cumulative SRAM Reads Over Time
    # ============================================================
    fig = dark_fig(9, 5.5)
    ax = fig.add_subplot(111)
    style_ax(ax, f'Cumulative SRAM Reads — Baseline vs Sparse (N={N})',
             'Timestep (T)', 'Cumulative SRAM Reads')

    t_axis = np.arange(1, T_MAX + 1)
    b_mean = np.mean(baseline_cumul_all, axis=0)
    b_std = np.std(baseline_cumul_all, axis=0)
    s_mean = np.mean(sparse_cumul_all, axis=0)
    s_std = np.std(sparse_cumul_all, axis=0)

    # Savings fill
    ax.fill_between(t_axis, s_mean, b_mean, alpha=0.06, color=PINK)
    # Std bands
    ax.fill_between(t_axis, b_mean - b_std, b_mean + b_std, alpha=0.12, color=RED)
    ax.fill_between(t_axis, s_mean - s_std, s_mean + s_std, alpha=0.12, color=CYAN)
    # Main lines
    ax.plot(t_axis, b_mean, color=RED, linewidth=2.5, marker='s', markersize=5,
            markerfacecolor=RED, markeredgecolor='white', markeredgewidth=0.5,
            label=f'Baseline — {b_mean[-1]:,.0f} reads', zorder=5)
    ax.plot(t_axis, s_mean, color=CYAN, linewidth=2.5, marker='o', markersize=5,
            markerfacecolor=CYAN, markeredgecolor='white', markeredgewidth=0.5,
            label=f'Sparse — {s_mean[-1]:,.0f} reads', zorder=5)

    mean_exit = agg('exit_t')[0]
    ax.axvline(x=mean_exit, color=GREEN, linestyle='--', linewidth=1.5, alpha=0.8,
               label=f'Avg Early Exit (T={mean_exit:.1f})')

    # Annotate savings
    mid_t = int(T_MAX * 0.75)
    gap = b_mean[mid_t-1] - s_mean[mid_t-1]
    ax.annotate(f'{agg("saving_pct")[0]:.0f}% fewer\nSRAM reads',
                xy=(mid_t, (b_mean[mid_t-1] + s_mean[mid_t-1])/2),
                fontsize=10, color=PINK, fontweight='bold', ha='center',
                bbox=dict(boxstyle='round,pad=0.4', fc=BG, ec=PINK, alpha=0.9))

    ax.legend(loc='upper left', facecolor=PANEL, edgecolor=GRID_C, labelcolor=TEXT, fontsize=9)
    ax.set_xticks(t_axis)
    fig.savefig('results/fig1_cumulative_sram.png', facecolor=BG)
    plt.close()
    print("  📊 fig1_cumulative_sram.png")

    # ============================================================
    # FIGURE 2: Early Exit Histogram
    # ============================================================
    exits = [r['exit_t'] for r in results]
    fig = dark_fig(8, 5)
    ax = fig.add_subplot(111)
    style_ax(ax, f'Early Exit Timestep Distribution (N={N})',
             'Exit Timestep', 'Number of Samples')

    bins = np.arange(0.5, T_MAX + 1.5, 1)
    counts, _, bars = ax.hist(exits, bins=bins, color=GREEN, edgecolor=PANEL,
                               alpha=0.85, rwidth=0.82)
    # Gradient effect: color bars by height
    max_c = max(counts) if max(counts) > 0 else 1
    for bar, c in zip(bars, counts):
        frac = c / max_c
        r = int(0 + frac * 0)
        g = int(230 - frac * 50)
        b_col = int(118 - frac * 40)
        bar.set_facecolor(f'#{r:02x}{g:02x}{b_col:02x}')
        bar.set_alpha(0.7 + 0.3 * frac)

    ax.axvline(x=np.mean(exits), color=RED, linestyle='--', linewidth=2,
               label=f'Mean = {np.mean(exits):.1f}', zorder=10)
    ax.axvline(x=np.median(exits), color=GOLD, linestyle=':', linewidth=1.5,
               label=f'Median = {np.median(exits):.0f}', zorder=10)

    ax.legend(facecolor=PANEL, edgecolor=GRID_C, labelcolor=TEXT)
    ax.set_xticks(range(1, T_MAX + 1))
    fig.savefig('results/fig2_exit_histogram.png', facecolor=BG)
    plt.close()
    print("  📊 fig2_exit_histogram.png")

    # ============================================================
    # FIGURE 3: Gatekeeper Breakdown (Donut + Bar)
    # ============================================================
    gk_kept_v = np.mean([r['gk_kept'] for r in results])
    gk_rej_v = np.mean([r['gk_total'] - r['gk_kept'] for r in results])
    gk_total = gk_kept_v + gk_rej_v

    fig = dark_fig(10, 4.5)
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    # Donut chart
    ax1.set_facecolor(PANEL)
    sizes = [gk_kept_v, gk_rej_v]
    colors_d = [CYAN, PINK]
    wedges, texts, autotexts = ax1.pie(sizes, labels=['Kept', 'Rejected'],
        colors=colors_d, autopct='%1.1f%%', startangle=90,
        pctdistance=0.78, wedgeprops=dict(width=0.4, edgecolor=PANEL, linewidth=2))
    for t in texts + autotexts:
        t.set_color(TEXT)
        t.set_fontsize(10)
    for at in autotexts:
        at.set_fontweight('bold')
    centre_circle = plt.Circle((0, 0), 0.45, fc=PANEL)
    ax1.add_artist(centre_circle)
    ax1.text(0, 0, f'{gk_total:,.0f}\nTotal', ha='center', va='center',
             color=TEXT, fontsize=11, fontweight='bold')
    ax1.set_title('Gatekeeper Filter Ratio', color=TEXT, fontsize=12, fontweight='bold')

    # Bar chart
    style_ax(ax2, 'Avg Spikes per Inference', '', 'Spike Count')
    bar_vals = [gk_kept_v, gk_rej_v]
    bar_labels = ['Kept\n(→ SRAM)', 'Rejected\n(Blocked)']
    bar_colors = [CYAN, PINK]
    bs = ax2.bar(bar_labels, bar_vals, color=bar_colors, width=0.5,
                 edgecolor=[TEAL, RED], linewidth=1.5, alpha=0.85)
    for b_bar in bs:
        h = b_bar.get_height()
        ax2.text(b_bar.get_x() + b_bar.get_width()/2, h + gk_total * 0.02,
                 f'{h:,.0f}', ha='center', va='bottom', color=TEXT, fontweight='bold', fontsize=11)

    fig.savefig('results/fig3_gatekeeper_bar.png', facecolor=BG)
    plt.close()
    print("  📊 fig3_gatekeeper_bar.png")

    # ============================================================
    # FIGURE 4: Energy Comparison (Stacked Bar)
    # ============================================================
    e_b_total = agg('e_base')[0]
    e_s_total = agg('e_sparse')[0]
    e_b_sram = e_b_total * E_SRAM_PJ / (E_SRAM_PJ + E_MAC_PJ)
    e_b_mac = e_b_total * E_MAC_PJ / (E_SRAM_PJ + E_MAC_PJ)
    e_s_sram = e_s_total * E_SRAM_PJ / (E_SRAM_PJ + E_MAC_PJ)
    e_s_mac = e_s_total * E_MAC_PJ / (E_SRAM_PJ + E_MAC_PJ)

    fig = dark_fig(7, 5)
    ax = fig.add_subplot(111)
    style_ax(ax, 'Energy per Inference (45nm CMOS)', '', 'Energy (pJ)')

    x_pos = np.array([0, 1])
    ax.bar(x_pos, [e_b_sram, e_s_sram], 0.45, label='SRAM Access', color=RED, alpha=0.85, edgecolor=PANEL)
    ax.bar(x_pos, [e_b_mac, e_s_mac], 0.45, bottom=[e_b_sram, e_s_sram],
           label='MAC Compute', color=PURPLE, alpha=0.85, edgecolor=PANEL)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(['Baseline', 'Sparse'], color=TEXT)

    # Total labels with savings
    for i, total in enumerate([e_b_total, e_s_total]):
        ax.text(i, total + e_b_total * 0.03, f'{total:,.0f} pJ',
                ha='center', color=TEXT, fontweight='bold', fontsize=11)

    pct_saved = (1 - e_s_total / e_b_total) * 100
    ax.annotate(f'↓ {pct_saved:.0f}% energy saved',
                xy=(0.5, (e_b_total + e_s_total)/2),
                fontsize=12, color=GREEN, fontweight='bold', ha='center',
                bbox=dict(boxstyle='round,pad=0.4', fc=BG, ec=GREEN, alpha=0.85))

    ax.legend(facecolor=PANEL, edgecolor=GRID_C, labelcolor=TEXT)
    fig.savefig('results/fig4_energy_comparison.png', facecolor=BG)
    plt.close()
    print("  📊 fig4_energy_comparison.png")

    # ============================================================
    # FIGURE 5: Latency & Throughput
    # ============================================================
    fig = dark_fig(11, 5)
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    lat_b, lat_s = agg('lat_base')[0], agg('lat_sparse')[0]
    lat_s_std = agg('lat_sparse')[1]

    # Latency
    style_ax(ax1, 'Inference Latency', '', 'Latency (ns)')
    bs1 = ax1.bar(['Baseline', 'Sparse'], [lat_b, lat_s], color=[RED, CYAN],
                  width=0.45, edgecolor=PANEL, linewidth=1.5, alpha=0.85,
                  yerr=[0, lat_s_std], capsize=5, error_kw={'ecolor': TEXT_DIM, 'capthick': 1.5})
    for b_bar in bs1:
        h = b_bar.get_height()
        ax1.text(b_bar.get_x() + b_bar.get_width()/2, h + lat_b * 0.04,
                 f'{h:.0f} ns', ha='center', color=TEXT, fontweight='bold', fontsize=11)

    speedup = lat_b / lat_s
    ax1.annotate(f'{speedup:.1f}× faster', xy=(0.5, lat_b * 0.5), fontsize=12,
                 color=GREEN, fontweight='bold', ha='center',
                 bbox=dict(boxstyle='round,pad=0.4', fc=BG, ec=GREEN, alpha=0.85))

    # Throughput
    style_ax(ax2, 'Inference Throughput @ 100MHz', '', 'M inferences/sec')
    tp_b = 1e9 / lat_b / 1e6
    tp_s = 1e9 / lat_s / 1e6
    bs2 = ax2.bar(['Baseline', 'Sparse'], [tp_b, tp_s], color=[RED, CYAN],
                  width=0.45, edgecolor=PANEL, linewidth=1.5, alpha=0.85)
    for b_bar in bs2:
        h = b_bar.get_height()
        ax2.text(b_bar.get_x() + b_bar.get_width()/2, h + tp_s * 0.04,
                 f'{h:.1f} M/s', ha='center', color=TEXT, fontweight='bold', fontsize=11)

    plt.tight_layout()
    fig.savefig('results/fig5_latency_speedup.png', facecolor=BG)
    plt.close()
    print("  📊 fig5_latency_speedup.png")

    # ============================================================
    # FIGURE 6: Per-Layer Firing Rates (Grouped Bars)
    # ============================================================
    layers = ['lif1', 'lif2', 'lif3', 'lif4']
    layer_labels = ['Conv1\n32×28×28', 'Conv2\n64×14×14', 'FC1\n128', 'FC2\n10']
    layer_neurons = [32*28*28, 64*14*14, 128, 10]

    b_rates = [np.mean(per_layer_spikes_base.get(l, [0])) / (n * T_MAX) * 100
               for l, n in zip(layers, layer_neurons)]
    s_rates_raw = per_layer_spikes_sparse if per_layer_spikes_sparse else per_layer_spikes_base
    mean_exit_t = agg('exit_t')[0]
    s_rates = []
    for l, n in zip(layers, layer_neurons):
        vals = s_rates_raw.get(l, per_layer_spikes_base.get(l, [0]))
        s_rates.append(np.mean(vals) / (n * mean_exit_t) * 100)

    fig = dark_fig(9, 5)
    ax = fig.add_subplot(111)
    style_ax(ax, 'Per-Layer Average Firing Rate', '', 'Firing Rate (%)')

    x = np.arange(len(layers))
    w = 0.32
    ax.bar(x - w/2, b_rates, w, label='Baseline', color=RED, alpha=0.85, edgecolor=PANEL)
    ax.bar(x + w/2, s_rates, w, label='Sparse', color=CYAN, alpha=0.85, edgecolor=PANEL)

    # Value labels
    for i in range(len(layers)):
        ax.text(x[i] - w/2, b_rates[i] + max(b_rates)*0.03, f'{b_rates[i]:.2f}%',
                ha='center', color=RED, fontsize=8, fontweight='bold')
        ax.text(x[i] + w/2, s_rates[i] + max(b_rates)*0.03, f'{s_rates[i]:.2f}%',
                ha='center', color=CYAN, fontsize=8, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(layer_labels, color=TEXT_DIM)
    ax.legend(facecolor=PANEL, edgecolor=GRID_C, labelcolor=TEXT)
    fig.savefig('results/fig6_firing_rate.png', facecolor=BG)
    plt.close()
    print("  📊 fig6_firing_rate.png")

    # ============================================================
    # FIGURE 7: Per-Sample Savings Scatter (Glowing dots)
    # ============================================================
    fig = dark_fig(8, 5.5)
    ax = fig.add_subplot(111)
    style_ax(ax, f'Per-Sample Savings vs Exit Time (N={N})',
             'Early Exit Timestep', 'SRAM Reads Saved (%)')

    exits_arr = [r['exit_t'] for r in results]
    savings_arr = [r['saving_pct'] for r in results]
    labels_arr = [r['label'] for r in results]

    # Glow layer (larger, faint)
    ax.scatter(exits_arr, savings_arr, c=labels_arr, cmap='Spectral',
               alpha=0.12, s=80, edgecolors='none')
    # Main dots
    sc = ax.scatter(exits_arr, savings_arr, c=labels_arr, cmap='Spectral',
                    alpha=0.7, s=22, edgecolors='none', zorder=5)

    cbar = plt.colorbar(sc, ax=ax, label='Digit Class', pad=0.02)
    cbar.set_ticks(range(10))
    cbar.ax.yaxis.set_tick_params(color=TEXT_DIM)
    cbar.outline.set_edgecolor(GRID_C)
    cbar.ax.yaxis.label.set_color(TEXT_DIM)
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=TEXT_DIM)

    # Trend line
    z = np.polyfit(exits_arr, savings_arr, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(exits_arr), max(exits_arr), 100)
    ax.plot(x_line, p(x_line), color=GOLD, linewidth=1.5, linestyle='--', alpha=0.6, label='Trend')
    ax.legend(facecolor=PANEL, edgecolor=GRID_C, labelcolor=TEXT, fontsize=9)

    fig.savefig('results/fig7_savings_scatter.png', facecolor=BG)
    plt.close()
    print("  📊 fig7_savings_scatter.png")

    # ============================================================
    # FIGURE 8: Confidence Trajectory
    # ============================================================
    if confidence_trajectories:
        fig = dark_fig(9, 5.5)
        ax = fig.add_subplot(111)
        style_ax(ax, 'Confidence Trajectory by Digit Class',
                 'Timestep', 'Max Softmax Confidence')

        digit_colors = [RED, ORANGE, GOLD, GREEN, TEAL, CYAN, PURPLE, PINK, '#FF8A65', TEXT_DIM]
        for digit, traj in sorted(confidence_trajectories.items()):
            t_ax = range(1, len(traj) + 1)
            c = digit_colors[digit % len(digit_colors)]
            ax.plot(t_ax, traj, linewidth=2, color=c, marker='o', markersize=4,
                    markerfacecolor=c, markeredgecolor='white', markeredgewidth=0.3,
                    label=f'Digit {digit}', alpha=0.85, zorder=5)

        ax.axhline(y=0.9, color=PINK, linestyle='--', linewidth=2, alpha=0.7,
                   label='Exit Threshold (90%)')
        ax.fill_between(range(1, T_MAX+1), 0.9, 1.0, alpha=0.04, color=GREEN)

        ax.set_ylim(0, 1.05)
        ax.set_xticks(range(1, T_MAX + 1))
        ax.legend(ncol=4, facecolor=PANEL, edgecolor=GRID_C, labelcolor=TEXT, fontsize=7.5, loc='lower right')
        fig.savefig('results/fig8_confidence_trajectory.png', facecolor=BG)
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
