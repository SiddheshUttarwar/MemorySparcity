import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_npz_files(data_dir: Path, max_samples: int):
    files = sorted(data_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found in {data_dir}")
    if max_samples > 0:
        files = files[:max_samples]
    return files


def load_data(files):
    xs = []
    ys = []
    for fp in files:
        d = np.load(fp)
        xs.append(d["x"].astype(np.float32))  # [T,2,H,W]
        ys.append(int(d["y"]) if "y" in d else -1)
    return np.stack(xs, axis=0), np.array(ys, dtype=np.int64)


def build_sample_images(x):
    # x: [N,T,2,H,W]
    on = x[:, :, 1].sum(axis=1)   # [N,H,W]
    off = x[:, :, 0].sum(axis=1)  # [N,H,W]
    signed = on - off
    return on, off, signed


def main():
    parser = argparse.ArgumentParser(description="Visualize preprocessed test dataset with images and graphs.")
    parser.add_argument("--data-dir", type=Path, default=Path("preprocessed_test"), help="Directory with preprocessed .npz test samples")
    parser.add_argument("--max-samples", type=int, default=500, help="Use first N samples (0 = all)")
    parser.add_argument("--num-display", type=int, default=12, help="How many sample images to display")
    parser.add_argument("--save-fig", type=Path, default=None, help="Optional path to save figure")
    args = parser.parse_args()

    files = load_npz_files(args.data_dir, args.max_samples)
    x, y = load_data(files)
    n, t, c, h, w = x.shape
    if c != 2:
        raise ValueError(f"Expected channels=2 (OFF/ON), got {c}")

    on_img, off_img, signed_img = build_sample_images(x)

    class_counts = np.bincount(y[y >= 0], minlength=10) if np.any(y >= 0) else np.zeros(10, dtype=np.int64)
    temporal_mean = x.sum(axis=(0, 2, 3, 4)) / max(1, n)
    on_total = float(x[:, :, 1].sum())
    off_total = float(x[:, :, 0].sum())
    on_ratio = on_total / max(on_total + off_total, 1e-12)
    sparsity = 1.0 - (np.count_nonzero(x, axis=(1, 2, 3, 4)) / np.prod(x.shape[1:]))

    print("=== Test Dataset Summary ===")
    print(f"Samples loaded: {n}")
    print(f"Sample shape: [T={t}, C={c}, H={h}, W={w}]")
    print(f"Class counts: {class_counts.tolist()}")
    print(f"Total ON events:  {on_total:.2f}")
    print(f"Total OFF events: {off_total:.2f}")
    print(f"ON ratio: {on_ratio:.4f}")
    print(f"Mean sparsity: {float(np.mean(sparsity)):.4f}")

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.0, 1.0, 1.2])

    # Graph 1: Class distribution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(np.arange(10), class_counts, color="tab:blue")
    ax1.set_title("Class Distribution")
    ax1.set_xlabel("Digit")
    ax1.set_ylabel("Count")
    ax1.set_xticks(np.arange(10))

    # Graph 2: Temporal event profile
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(np.arange(t), temporal_mean, marker="o", color="tab:orange")
    ax2.set_title("Temporal Activity")
    ax2.set_xlabel("Time Bin")
    ax2.set_ylabel("Mean Events / Sample")

    # Graph 3: ON/OFF totals
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.bar(["OFF", "ON"], [off_total, on_total], color=["tab:red", "tab:green"])
    ax3.set_title("Polarity Totals")
    ax3.set_ylabel("Event Count")

    # Graph 4: Sparsity distribution
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.hist(sparsity, bins=25, color="tab:purple", edgecolor="black", alpha=0.8)
    ax4.set_title("Sparsity Distribution")
    ax4.set_xlabel("Sparsity")
    ax4.set_ylabel("Samples")

    # Image panel: signed event images for samples
    m = min(args.num_display, n, 8)
    for i in range(m):
        ax = fig.add_subplot(gs[1 + i // 4, i % 4])
        im = signed_img[i]
        vmax = np.max(np.abs(im)) + 1e-6
        ax.imshow(im, cmap="seismic", vmin=-vmax, vmax=vmax)
        title = f"#{i}"
        if y[i] >= 0:
            title += f" (y={y[i]})"
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("Preprocessed Test Dataset Visualization", fontsize=14)
    fig.tight_layout()

    if args.save_fig is not None:
        args.save_fig.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save_fig, dpi=160, bbox_inches="tight")
        print(f"Saved figure: {args.save_fig}")

    plt.show()


if __name__ == "__main__":
    main()
