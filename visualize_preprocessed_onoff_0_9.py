import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def label_from_file(fp: Path):
    # Preferred: read label from file payload.
    with np.load(fp) as d:
        if "y" in d:
            return int(d["y"])
    # Fallback: parse from filename pattern *_labelX_*
    name = fp.name
    marker = "_label"
    i = name.find(marker)
    if i >= 0 and i + len(marker) < len(name):
        c = name[i + len(marker)]
        if c.isdigit():
            return int(c)
    return -1


def main():
    parser = argparse.ArgumentParser(description="Visualize aggregated ON/OFF maps for digits 0-9 from preprocessed .npz data.")
    parser.add_argument("--data-dir", type=Path, default=Path("preprocessed_train"), help="Directory with preprocessed .npz files")
    parser.add_argument("--samples-per-digit", type=int, default=200, help="Samples per digit (0 = all)")
    parser.add_argument("--save-fig", type=Path, default=Path("reports/preprocessed_onoff_0_9.png"))
    args = parser.parse_args()

    files = sorted(args.data_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found in {args.data_dir}")

    by_digit = {d: [] for d in range(10)}
    for fp in files:
        y = label_from_file(fp)
        if 0 <= y <= 9:
            by_digit[y].append(fp)

    for d in range(10):
        if args.samples_per_digit > 0:
            by_digit[d] = by_digit[d][: args.samples_per_digit]

    on_maps = {}
    off_maps = {}
    used = {}

    for d in range(10):
        selected = by_digit[d]
        used[d] = len(selected)
        on_sum = None
        off_sum = None

        for fp in selected:
            with np.load(fp) as data:
                x = data["x"].astype(np.float32)  # [T,2,H,W]
            off = x[:, 0].sum(axis=0)  # [H,W]
            on = x[:, 1].sum(axis=0)   # [H,W]
            if on_sum is None:
                on_sum = np.zeros_like(on, dtype=np.float64)
                off_sum = np.zeros_like(off, dtype=np.float64)
            on_sum += on
            off_sum += off

        if on_sum is None:
            # No samples for this digit, keep zero map
            on_maps[d] = np.zeros((34, 34), dtype=np.float32)
            off_maps[d] = np.zeros((34, 34), dtype=np.float32)
        else:
            n = max(1, len(selected))
            on_maps[d] = (on_sum / n).astype(np.float32)
            off_maps[d] = (off_sum / n).astype(np.float32)

    print("Samples used per digit (preprocessed):")
    for d in range(10):
        print(f"  {d}: {used[d]}")

    vmax_on = max(float(np.max(on_maps[d])) for d in range(10))
    vmax_off = max(float(np.max(off_maps[d])) for d in range(10))

    fig, axes = plt.subplots(2, 10, figsize=(22, 5), constrained_layout=True)
    for d in range(10):
        ax_on = axes[0, d]
        ax_off = axes[1, d]
        im_on = ax_on.imshow(on_maps[d], cmap="Blues", vmin=0, vmax=vmax_on + 1e-12)
        im_off = ax_off.imshow(off_maps[d], cmap="Reds", vmin=0, vmax=vmax_off + 1e-12)
        ax_on.set_title(f"{d} ON")
        ax_off.set_title(f"{d} OFF")
        ax_on.set_xticks([])
        ax_on.set_yticks([])
        ax_off.set_xticks([])
        ax_off.set_yticks([])
        if d == 0:
            ax_on.set_ylabel("ON")
            ax_off.set_ylabel("OFF")

    cbar1 = fig.colorbar(im_on, ax=axes[0, :], fraction=0.02, pad=0.01)
    cbar1.set_label("Mean ON events / pixel")
    cbar2 = fig.colorbar(im_off, ax=axes[1, :], fraction=0.02, pad=0.01)
    cbar2.set_label("Mean OFF events / pixel")
    fig.suptitle(f"Preprocessed ON/OFF Maps by Digit (0-9) | {args.data_dir.name}")

    args.save_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.save_fig, dpi=170, bbox_inches="tight")
    print(f"Saved figure: {args.save_fig}")
    plt.show()


if __name__ == "__main__":
    main()

