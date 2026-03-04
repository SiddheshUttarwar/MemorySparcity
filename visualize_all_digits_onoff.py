import argparse
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def decode_events(raw_bytes: bytes):
    stream = np.frombuffer(raw_bytes, dtype=np.uint8)
    if stream.size % 5 != 0:
        raise ValueError("Corrupt sample: byte length not divisible by 5.")
    x = stream[0::5].astype(np.int16)
    y = stream[1::5].astype(np.int16)
    b2 = stream[2::5].astype(np.uint32)
    p = (b2 >> 7).astype(np.uint8)  # 0=OFF, 1=ON
    return x, y, p


def accumulate_on_off(x, y, p, width=34, height=34):
    on = np.zeros((height, width), dtype=np.float32)
    off = np.zeros((height, width), dtype=np.float32)
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    x = x[valid]
    y = y[valid]
    p = p[valid]
    on_mask = p == 1
    off_mask = ~on_mask
    np.add.at(on, (y[on_mask], x[on_mask]), 1.0)
    np.add.at(off, (y[off_mask], x[off_mask]), 1.0)
    return on, off


def collect_members(zip_path: Path):
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = [n for n in zf.namelist() if n.endswith(".bin")]
    by_label = {k: [] for k in range(10)}
    for m in members:
        parts = m.split("/")
        if len(parts) >= 3 and parts[1].isdigit():
            lbl = int(parts[1])
            if 0 <= lbl <= 9:
                by_label[lbl].append(m)
    for k in by_label:
        by_label[k].sort()
    return by_label


def main():
    parser = argparse.ArgumentParser(description="Visualize aggregated ON/OFF event maps for digits 0-9.")
    parser.add_argument("--zip-path", type=Path, default=Path("Train.zip"), help="Path to Train.zip or Test.zip")
    parser.add_argument("--samples-per-digit", type=int, default=200, help="How many samples to aggregate per digit (0 = all)")
    parser.add_argument("--save-fig", type=Path, default=Path("reports/all_digits_onoff.png"), help="Path to save figure")
    args = parser.parse_args()

    by_label = collect_members(args.zip_path)
    on_maps = {}
    off_maps = {}
    used_counts = {}

    with zipfile.ZipFile(args.zip_path, "r") as zf:
        for digit in range(10):
            files = by_label[digit]
            if args.samples_per_digit > 0:
                files = files[: args.samples_per_digit]
            used_counts[digit] = len(files)

            on_sum = np.zeros((34, 34), dtype=np.float32)
            off_sum = np.zeros((34, 34), dtype=np.float32)
            for m in files:
                raw = zf.read(m)
                x, y, p = decode_events(raw)
                on, off = accumulate_on_off(x, y, p, width=34, height=34)
                on_sum += on
                off_sum += off

            n = max(1, len(files))
            on_maps[digit] = on_sum / n
            off_maps[digit] = off_sum / n

    print("Samples used per digit:")
    for d in range(10):
        print(f"  {d}: {used_counts[d]}")

    fig, axes = plt.subplots(2, 10, figsize=(22, 5), constrained_layout=True)
    vmax_on = max(float(np.max(on_maps[d])) for d in range(10))
    vmax_off = max(float(np.max(off_maps[d])) for d in range(10))

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
    fig.suptitle(f"Aggregated ON/OFF Maps by Digit (0-9) | {args.zip_path.name}")

    args.save_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.save_fig, dpi=170, bbox_inches="tight")
    print(f"Saved figure: {args.save_fig}")
    plt.show()


if __name__ == "__main__":
    main()

