import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib as mpl

# ----------------------------
# Architecture-style animation
# ----------------------------
# Simple SNN layer engine:
#   input spikes -> (dense scan OR sparse controller) -> weight SRAM -> MAC -> neuron accum
#
# We visualize two lanes:
#   Top: Dense baseline (scan all K rows, gate compute)
#   Bottom: Sparse controller (only active rows, only nonzero synapses)

# Parameters
K = 32          # presynaptic neurons (kept small to visualize)
N = 64          # postsynaptic neurons
w_sparsity = 0.85
spike_rate = 0.12
T = 25
seed = 3
rng = np.random.default_rng(seed)

# Weight nonzero mask
W_nz = rng.random((K, N)) > w_sparsity

# Spike trains
S = rng.random((T, K)) < spike_rate

# Per-timestep stats
dense_reads = np.zeros(T, dtype=int)
sparse_reads = np.zeros(T, dtype=int)

for t in range(T):
    active = np.where(S[t])[0]
    dense_reads[t] = K * N              # dense architecture "touches" whole matrix per tick (worst-case scan)
    # A more realistic dense baseline: scan all rows (read row headers) and read full row if spike.
    # But for visualization, we keep a strong contrast: dense touches all weights each tick.
    sparse_reads[t] = int(W_nz[active].sum()) if active.size else 0

dense_cum = np.cumsum(dense_reads)
sparse_cum = np.cumsum(sparse_reads)

# --- Figure layout
fig = plt.figure(figsize=(13, 7))
gs = fig.add_gridspec(2, 3, width_ratios=[1.2, 1.6, 1.8], height_ratios=[1.0, 1.0], wspace=0.25, hspace=0.35)

ax_spike = fig.add_subplot(gs[:, 0])   # spike vector
ax_arch  = fig.add_subplot(gs[:, 1])   # architecture schematic
ax_plot  = fig.add_subplot(gs[:, 2])   # reads plot

# Spike vector heatmap
sp_img = ax_spike.imshow(np.zeros((K, 1)), aspect='auto', interpolation='nearest')
ax_spike.set_title("Input spikes s[t] (K×1)")
ax_spike.set_xticks([])
ax_spike.set_yticks([0, K//2, K-1])
ax_spike.set_ylabel("Presynaptic index")

# Plot reads
x = np.arange(T)
ax_plot.set_title("Weight-memory reads (proxy)")
ax_plot.set_xlabel("timestep")
ax_plot.set_ylabel("reads")
ax_plot.grid(True, alpha=0.3)
line_d, = ax_plot.plot([], [], label="Dense reads/step")
line_s, = ax_plot.plot([], [], label="Sparse reads/step")
line_dc, = ax_plot.plot([], [], linestyle="--", label="Dense cumulative")
line_sc, = ax_plot.plot([], [], linestyle="--", label="Sparse cumulative")
marker = ax_plot.axvline(0, linestyle=":", linewidth=2)
ax_plot.legend(loc="upper left")

ax_plot.set_xlim(0, T-1)
ymax = max(dense_reads.max(), sparse_reads.max(), dense_cum.max(), sparse_cum.max())
ax_plot.set_ylim(0, ymax * 1.05)

# Architecture schematic: draw blocks
ax_arch.axis("off")
ax_arch.set_title("SNN layer: Dense scan vs Sparse controller")

def block(ax, xy, w, h, text, fc="white"):
    p = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.02",
                       linewidth=1.5, edgecolor="black", facecolor=fc)
    ax.add_patch(p)
    ax.text(xy[0]+w/2, xy[1]+h/2, text, ha="center", va="center", fontsize=10)
    return p

def arrow(ax, p1, p2):
    a = FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=12, linewidth=1.4, color="black")
    ax.add_patch(a)
    return a

# Normalized coordinates in ax_arch
# Two lanes
dense_y = 0.68
sparse_y = 0.24
bh = 0.18
bw = 0.22

# Blocks for dense lane
b_in_d   = block(ax_arch, (0.02, dense_y), bw, bh, "Spike buffer\ns[t]")
b_ctrl_d = block(ax_arch, (0.28, dense_y), bw, bh, "Dense scan\ncontroller\n(scan all K)")
b_mem_d  = block(ax_arch, (0.54, dense_y), bw, bh, "Weight SRAM\n(W)")
b_mac_d  = block(ax_arch, (0.80, dense_y), bw, bh, "MAC + Accum\n(V update)")

# Blocks for sparse lane
b_in_s   = block(ax_arch, (0.02, sparse_y), bw, bh, "Spike buffer\ns[t]")
b_ctrl_s = block(ax_arch, (0.28, sparse_y), bw, bh, "Sparse controller\n(encode active k)\n+ FSM")
b_mem_s  = block(ax_arch, (0.54, sparse_y), bw, bh, "Sparse weight\nstorage\n(CSR/BCSR)")
b_mac_s  = block(ax_arch, (0.80, sparse_y), bw, bh, "MAC + Accum\n(V update)")

# Arrows
arrow(ax_arch, (0.24, dense_y+bh/2), (0.28, dense_y+bh/2))
arrow(ax_arch, (0.50, dense_y+bh/2), (0.54, dense_y+bh/2))
arrow(ax_arch, (0.76, dense_y+bh/2), (0.80, dense_y+bh/2))

arrow(ax_arch, (0.24, sparse_y+bh/2), (0.28, sparse_y+bh/2))
arrow(ax_arch, (0.50, sparse_y+bh/2), (0.54, sparse_y+bh/2))
arrow(ax_arch, (0.76, sparse_y+bh/2), (0.80, sparse_y+bh/2))

# Highlight overlays that will be updated each frame
hi_d = FancyBboxPatch((0.54, dense_y), bw, bh, boxstyle="round,pad=0.02,rounding_size=0.02",
                      linewidth=3, edgecolor="black", facecolor="none", alpha=0.0)
hi_s = FancyBboxPatch((0.54, sparse_y), bw, bh, boxstyle="round,pad=0.02,rounding_size=0.02",
                      linewidth=3, edgecolor="black", facecolor="none", alpha=0.0)
ax_arch.add_patch(hi_d)
ax_arch.add_patch(hi_s)

txt_stats = ax_arch.text(0.02, 0.02, "", fontsize=10, va="bottom")

# Helper: compute sparse reads per active row for annotation
def sparse_breakdown(t):
    active = np.where(S[t])[0]
    if active.size == 0:
        return active, []
    nz_per_row = W_nz[active].sum(axis=1).astype(int)
    return active, nz_per_row

def update(t):
    # update spike column
    sp_img.set_data(S[t].astype(float).reshape(K, 1))

    # plot lines
    line_d.set_data(x[:t+1], dense_reads[:t+1])
    line_s.set_data(x[:t+1], sparse_reads[:t+1])
    line_dc.set_data(x[:t+1], dense_cum[:t+1])
    line_sc.set_data(x[:t+1], sparse_cum[:t+1])
    marker.set_xdata([t, t])

    # update highlights: flash memory blocks when reads happen
    # Dense always reads in this simplified baseline
    hi_d.set_alpha(0.8)
    hi_s.set_alpha(0.8 if sparse_reads[t] > 0 else 0.15)

    # annotate stats
    active, nz_per = sparse_breakdown(t)
    if active.size == 0:
        detail = "active k: none"
    else:
        # show up to 6 active indices
        idx_show = ", ".join(map(str, active[:6]))
        if active.size > 6:
            idx_show += ", …"
        nz_show = ", ".join(map(str, nz_per[:6]))
        if active.size > 6:
            nz_show += ", …"
        detail = f"active k: [{idx_show}]\nnonzeros per active row: [{nz_show}]"

    txt_stats.set_text(
        f"t = {t}\n"
        f"Dense: reads K×N = {K}×{N} = {dense_reads[t]:,}\n"
        f"Sparse: reads = {sparse_reads[t]:,}\n"
        f"{detail}"
    )

    return sp_img, line_d, line_s, line_dc, line_sc, marker, hi_d, hi_s, txt_stats

anim = FuncAnimation(fig, update, frames=T, interval=450, blit=False)

out_path = "/snn_arch_memory_sparsity.gif"
anim.save(out_path, writer="pillow", dpi=95)
plt.close(fig)

out_path