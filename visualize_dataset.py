import argparse
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

"""
N-MNIST dataset visualizer.

This script reads one N-MNIST sample from Train.zip or Test.zip and visualizes
its event stream with three plots.

Dataset/event format used here:
- Each sample is a .bin file inside zip folders like Train/<digit>/xxxxx.bin.
- Each event is 5 bytes (40 bits):
  - byte 1: x address
  - byte 2: y address
  - byte 3 MSB: polarity p (0=OFF, 1=ON)
  - byte 3 lower 7 bits + byte 4 + byte 5: timestamp (microseconds)

Meaning of ON and OFF:
- ON event (p=1): local brightness increased at that pixel.
- OFF event (p=0): local brightness decreased at that pixel.
- Significance: ON/OFF together preserve contrast-change direction, which helps
  motion/edge interpretation and improves event-based model debugging/training.

Plots generated:
1) Event raster (time vs y):
   - ON and OFF events are shown as colored points over time.
   - Why useful: validates temporal spike behavior and polarity timing patterns.

2) ON event count image:
   - 2D histogram of ON events per pixel.
   - Why useful: shows where positive contrast changes are concentrated.

3) OFF event count image:
   - 2D histogram of OFF events per pixel.
   - Why useful: shows where negative contrast changes are concentrated.

Combined significance:
- Raster gives temporal behavior.
- ON/OFF images give spatial behavior.
- Together they provide a compact sanity check before SNN feature extraction,
  encoding experiments, or training.

Usage examples:
- python visualize_dataset.py
- python visualize_dataset.py --zip-path Train.zip --label 3 --sample-index 10
- python visualize_dataset.py --zip-path Test.zip --label 7 --sample-index 5
"""


def decode_events(raw_bytes: bytes):
    """Decode N-MNIST 5-byte event records into x, y, polarity, timestamp arrays."""
    stream = np.frombuffer(raw_bytes, dtype=np.uint8)
    if stream.size % 5 != 0:
        raise ValueError("Corrupt sample: byte length is not divisible by 5.")

    x = stream[0::5].astype(np.int16)
    y = stream[1::5].astype(np.int16)
    b2 = stream[2::5].astype(np.uint32)
    p = (b2 >> 7).astype(np.uint8)  # 0=OFF, 1=ON
    ts = ((b2 & 0x7F) << 16) | (stream[3::5].astype(np.uint32) << 8) | stream[4::5].astype(np.uint32)
    return x, y, p, ts


def pick_sample_from_zip(zip_path: Path, label: str, sample_index: int):
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = [n for n in zf.namelist() if n.endswith(".bin") and f"/{label}/" in n]
        if not members:
            raise FileNotFoundError(f"No .bin files found for label '{label}' in {zip_path}")
        members.sort()
        if sample_index < 0 or sample_index >= len(members):
            raise IndexError(f"sample-index must be in [0, {len(members) - 1}] for label {label}")
        member = members[sample_index]
        raw = zf.read(member)
    return member, raw


def accumulate_frame(x, y, p, width=34, height=34):
    on = np.zeros((height, width), dtype=np.int32)
    off = np.zeros((height, width), dtype=np.int32)

    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    x = x[valid]
    y = y[valid]
    p = p[valid]

    on_mask = p == 1
    off_mask = ~on_mask

    np.add.at(on, (y[on_mask], x[on_mask]), 1)
    np.add.at(off, (y[off_mask], x[off_mask]), 1)

    return on, off


def plot_sample(x, y, p, ts, title):
    on, off = accumulate_frame(x, y, p)
    start_t = float(ts.min()) if len(ts) else 0.0
    rel_t = (ts.astype(np.float64) - start_t) / 1e3  # ms

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    ax0, ax1, ax2 = axes

    ax0.scatter(rel_t[p == 1], y[p == 1], s=1.5, c="tab:blue", label="ON", alpha=0.7)
    ax0.scatter(rel_t[p == 0], y[p == 0], s=1.5, c="tab:red", label="OFF", alpha=0.7)
    ax0.set_title("Event Raster (time vs y)")
    ax0.set_xlabel("Time (ms)")
    ax0.set_ylabel("Y pixel")
    ax0.invert_yaxis()
    ax0.legend(loc="best")

    im1 = ax1.imshow(on, cmap="Blues")
    ax1.set_title("ON Event Count")
    ax1.set_xlabel("X pixel")
    ax1.set_ylabel("Y pixel")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    im2 = ax2.imshow(off, cmap="Reds")
    ax2.set_title("OFF Event Count")
    ax2.set_xlabel("X pixel")
    ax2.set_ylabel("Y pixel")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    fig.suptitle(title)
    fig.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize one N-MNIST sample from Train.zip or Test.zip.")
    parser.add_argument("--zip-path", type=Path, default=Path("Train.zip"), help="Path to zip archive (default: Train.zip)")
    parser.add_argument("--label", type=str, default="0", help="Digit class folder inside archive (0-9).")
    parser.add_argument("--sample-index", type=int, default=0, help="Index into sorted .bin files for that label.")
    args = parser.parse_args()

    member, raw = pick_sample_from_zip(args.zip_path, args.label, args.sample_index)
    x, y, p, ts = decode_events(raw)

    title = f"{args.zip_path.name} | {member} | events={len(x)}"
    print(title)
    if len(ts):
        print(f"time range: {int(ts.min())}us -> {int(ts.max())}us")
    plot_sample(x, y, p, ts, title)


if __name__ == "__main__":
    main()
