import argparse
import json
import zipfile
from pathlib import Path

import numpy as np


"""
N-MNIST preprocessing utility.

What this script does:
1) Reads event samples from Train.zip/Test.zip.
2) Decodes raw 5-byte events into x, y, polarity, timestamp.
3) Applies basic cleaning:
   - removes out-of-bounds pixels
   - optional per-pixel denoising via minimum event count threshold
4) Converts events into a fixed tensor by time-binning:
   - output shape: [time_bins, 2, height, width]
   - channel 0 = OFF events, channel 1 = ON events
5) Optionally clips and normalizes tensor values.
6) Saves one compressed .npz per sample + a manifest.json.

Example:
python preprocess_dataset.py --zip-path Train.zip --output-dir preprocessed_train
"""


def decode_events(raw_bytes: bytes):
    stream = np.frombuffer(raw_bytes, dtype=np.uint8)
    if stream.size % 5 != 0:
        raise ValueError("Corrupt sample: byte length is not divisible by 5.")

    x = stream[0::5].astype(np.int16)
    y = stream[1::5].astype(np.int16)
    b2 = stream[2::5].astype(np.uint32)
    p = (b2 >> 7).astype(np.uint8)  # 0=OFF, 1=ON
    ts = ((b2 & 0x7F) << 16) | (stream[3::5].astype(np.uint32) << 8) | stream[4::5].astype(np.uint32)
    return x, y, p, ts


def filter_events(x, y, p, ts, width, height):
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    return x[valid], y[valid], p[valid], ts[valid]


def denoise_by_pixel_count(x, y, p, ts, min_events_per_pixel):
    if min_events_per_pixel <= 1:
        return x, y, p, ts

    # Keep only pixels that fire at least `min_events_per_pixel` times in a sample.
    coords = np.stack([y, x], axis=1)
    uniq, counts = np.unique(coords, axis=0, return_counts=True)
    keep_pix = uniq[counts >= min_events_per_pixel]
    if keep_pix.size == 0:
        return x[:0], y[:0], p[:0], ts[:0]

    keys = y.astype(np.int32) * 1000 + x.astype(np.int32)
    keep_keys = keep_pix[:, 0].astype(np.int32) * 1000 + keep_pix[:, 1].astype(np.int32)
    keep_mask = np.isin(keys, keep_keys)
    return x[keep_mask], y[keep_mask], p[keep_mask], ts[keep_mask]


def events_to_tensor(x, y, p, ts, time_bins, width, height):
    tensor = np.zeros((time_bins, 2, height, width), dtype=np.float32)
    if len(ts) == 0:
        return tensor

    t0 = ts.min()
    t1 = ts.max()
    if t1 == t0:
        bins = np.zeros_like(ts, dtype=np.int32)
    else:
        rel = (ts.astype(np.float64) - float(t0)) / float(t1 - t0)
        bins = np.clip((rel * (time_bins - 1)).astype(np.int32), 0, time_bins - 1)

    # channel: OFF=0, ON=1
    np.add.at(tensor, (bins, p.astype(np.int32), y.astype(np.int32), x.astype(np.int32)), 1.0)
    return tensor


def normalize_tensor(tensor, mode):
    if mode == "none":
        return tensor
    if mode == "max":
        m = float(tensor.max())
        return tensor / m if m > 0 else tensor
    if mode == "sum":
        s = float(tensor.sum())
        return tensor / s if s > 0 else tensor
    raise ValueError(f"Unsupported normalize mode: {mode}")


def iter_bin_members(zip_path: Path):
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = [n for n in zf.namelist() if n.endswith(".bin")]
    members.sort()
    return members


def label_from_member(member_name: str):
    # Expected: Train/3/00010.bin or Test/9/00123.bin
    parts = member_name.split("/")
    if len(parts) < 3:
        raise ValueError(f"Unexpected member path format: {member_name}")
    return int(parts[1])


def preprocess_archive(
    zip_path: Path,
    output_dir: Path,
    time_bins: int,
    width: int,
    height: int,
    min_events_per_pixel: int,
    clip_count: float,
    normalize: str,
    max_samples: int,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    members = iter_bin_members(zip_path)
    if max_samples > 0:
        members = members[:max_samples]

    manifest = {
        "zip_path": str(zip_path),
        "num_samples": len(members),
        "time_bins": time_bins,
        "shape": [time_bins, 2, height, width],
        "normalize": normalize,
        "min_events_per_pixel": min_events_per_pixel,
        "clip_count": clip_count,
        "samples": [],
    }

    with zipfile.ZipFile(zip_path, "r") as zf:
        for i, member in enumerate(members):
            raw = zf.read(member)
            x, y, p, ts = decode_events(raw)
            x, y, p, ts = filter_events(x, y, p, ts, width=width, height=height)
            x, y, p, ts = denoise_by_pixel_count(x, y, p, ts, min_events_per_pixel=min_events_per_pixel)
            tensor = events_to_tensor(x, y, p, ts, time_bins=time_bins, width=width, height=height)
            if clip_count > 0:
                tensor = np.clip(tensor, 0, clip_count)
            tensor = normalize_tensor(tensor, mode=normalize)

            label = label_from_member(member)
            stem = Path(member).stem
            sample_out = output_dir / f"{i:05d}_label{label}_{stem}.npz"
            np.savez_compressed(sample_out, x=tensor.astype(np.float32), y=np.int64(label))

            if len(ts) > 0:
                tmin = int(ts.min())
                tmax = int(ts.max())
            else:
                tmin = 0
                tmax = 0

            manifest["samples"].append(
                {
                    "index": i,
                    "member": member,
                    "label": label,
                    "output": sample_out.name,
                    "num_events_after_filter": int(len(ts)),
                    "time_us_min": tmin,
                    "time_us_max": tmax,
                }
            )

            if (i + 1) % 500 == 0 or i == len(members) - 1:
                print(f"Processed {i + 1}/{len(members)} samples")

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote manifest: {manifest_path}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess N-MNIST events into fixed tensors.")
    parser.add_argument("--zip-path", type=Path, required=True, help="Path to Train.zip or Test.zip")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to write processed .npz samples")
    parser.add_argument("--time-bins", type=int, default=10, help="Number of temporal bins")
    parser.add_argument("--width", type=int, default=34, help="Sensor width")
    parser.add_argument("--height", type=int, default=34, help="Sensor height")
    parser.add_argument(
        "--min-events-per-pixel",
        type=int,
        default=1,
        help="Keep only pixels with at least this many events in a sample",
    )
    parser.add_argument(
        "--clip-count",
        type=float,
        default=0.0,
        help="If > 0, clip per-bin event counts to this max value",
    )
    parser.add_argument(
        "--normalize",
        type=str,
        choices=["none", "max", "sum"],
        default="none",
        help="Tensor normalization mode",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="If > 0, process only first N samples (for quick experiments)",
    )
    args = parser.parse_args()

    preprocess_archive(
        zip_path=args.zip_path,
        output_dir=args.output_dir,
        time_bins=args.time_bins,
        width=args.width,
        height=args.height,
        min_events_per_pixel=args.min_events_per_pixel,
        clip_count=args.clip_count,
        normalize=args.normalize,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()

