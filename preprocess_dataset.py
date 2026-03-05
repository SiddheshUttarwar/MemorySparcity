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

import numpy as np
from scipy.ndimage import rotate
import os
import zipfile
import multiprocessing as mp
from functools import partial
import time
import math

class NeuromorphicPreprocessor:
    def __init__(self, target_size=(28, 28), original_size=(34, 34), time_bins=20):
        self.target_size = target_size
        self.original_size = original_size
        self.time_bins = time_bins

    def crop_events(self, x, y, p, t):
        offset_x = (self.original_size[0] - self.target_size[0]) // 2
        offset_y = (self.original_size[1] - self.target_size[1]) // 2
        
        mask = (x >= offset_x) & (x < offset_x + self.target_size[0]) & \
               (y >= offset_y) & (y < offset_y + self.target_size[1])
        return x[mask] - offset_x, y[mask] - offset_y, p[mask], t[mask]

    def native_event_to_tensor(self, x, y, p, t):
        spike_tensor = np.zeros((self.time_bins, 2, self.target_size[1], self.target_size[0]), dtype=np.bool_)
        if len(t) == 0:
            return spike_tensor
            
        t_min, t_max = t.min(), t.max()
        if t_max > t_min:
            t_norm = (t - t_min) / (t_max - t_min)
            t_binned = np.clip((t_norm * self.time_bins).astype(int), 0, self.time_bins - 1)
        else:
            t_binned = np.zeros_like(t, dtype=int)
            
        np.add.at(spike_tensor, (t_binned, p, y, x), True)
        return spike_tensor
        
    def one_hot_encode(self, label, num_classes=10):
        one_hot = np.zeros(num_classes, dtype=np.float32)
        one_hot[int(label)] = 1.0
        return one_hot

def read_nmnist_events(data):
    events = np.frombuffer(data, dtype=np.uint8)
    events = events.reshape(-1, 5)
    x, y = events[:, 0], events[:, 1]
    p = (events[:, 2] > 0).astype(int)
    t = (events[:, 3].astype(np.uint32) << 8) | events[:, 4].astype(np.uint32)
    return x, y, p, t

def process_file_chunk(file_infos_chunk, zip_path, out_dir):
    """ Processes a small chunk of zip entries to prevent multiprocessing hangs """
    processed = 0
    # Open local zipfile ref per worker
    with zipfile.ZipFile(zip_path, 'r') as z:
        processor = NeuromorphicPreprocessor(target_size=(28, 28), time_bins=20)
        for zip_info_name, dataset_name in file_infos_chunk:
            if not zip_info_name.endswith('.bin'):
                continue
                
            parts = zip_info_name.split('/')
            if len(parts) < 3:
                continue
                
            digit = int(parts[1])
            base_name = os.path.splitext(os.path.basename(zip_info_name))[0]
            out_file_path = os.path.join(out_dir, f"{dataset_name}_{digit}_{base_name}.npz")
            
            if os.path.exists(out_file_path):
                processed += 1
                continue # Already processed

            # Decode
            data = z.read(zip_info_name)
            x, y, p, t = read_nmnist_events(data)
            x_crop, y_crop, p_crop, t_crop = processor.crop_events(x, y, p, t)
            
            result_tensor = processor.native_event_to_tensor(x_crop, y_crop, p_crop, t_crop)
            label_one_hot = processor.one_hot_encode(digit)
            
            np.savez_compressed(out_file_path, data=result_tensor, label=label_one_hot, digit=digit)
            processed += 1
            
    return processed

def main():
    encoding_type = 'native'
    out_dir = f'preprocessed_data_{encoding_type}'
    os.makedirs(out_dir, exist_ok=True)
    
    datasets = [
        ('Test.zip', 'test'),
        ('Train.zip', 'train')
    ]
    
    processor_cores = max(1, mp.cpu_count() - 2)
    print(f"Starting FULL Dataset preprocessing natively with {processor_cores} cores. Saving to: {out_dir}")
    
    total_processed = 0
    start_time = time.time()
    
    for zip_path, dataset_name in datasets:
        if not os.path.exists(zip_path):
            print(f"WARNING: {zip_path} not found.")
            continue
            
        with zipfile.ZipFile(zip_path, 'r') as z:
            all_files = z.namelist()
            
        file_infos = [(f, dataset_name) for f in all_files if f.endswith('.bin')]
        print(f"Found {len(file_infos)} exact target files in {zip_path}")
        
        if not file_infos:
            continue
            
        # We partition the 60k/10k datasets into manageable chunks of 500 files each for the workers.
        chunk_size = 500
        num_chunks = math.ceil(len(file_infos) / chunk_size)
        chunks = [file_infos[i * chunk_size:(i + 1) * chunk_size] for i in range(num_chunks)]
        
        print(f"Divided {dataset_name} into {len(chunks)} processor chunks. Working...")

        worker_func = partial(process_file_chunk, zip_path=zip_path, out_dir=out_dir)
        
        processed_count = 0
        with mp.Pool(processes=processor_cores) as pool:
            for idx, c_res in enumerate(pool.imap_unordered(worker_func, chunks)):
                processed_count += c_res
                print(f" [{dataset_name}] Progress: {processed_count}/{len(file_infos)} files saved...")
            
        total_processed += processed_count
        print(f"Dataset {dataset_name} universally finished. Saved {processed_count} native .npz tensors.")
        
    print(f"\nAll completely done! Natively built {total_processed} files in {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    # Workaround for Windows Multiprocessing Fork Behavior
    mp.freeze_support()
    main()
