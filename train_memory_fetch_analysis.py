import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from LIF import LIFLayer
from SRAM import SRAMWeightMemory


LABEL_REGEX = re.compile(r"_label(\d)_")


def infer_label_from_filename(path: Path):
    m = LABEL_REGEX.search(path.name)
    return int(m.group(1)) if m else None


class NpzStreamDataset:
    def __init__(self, data_dir: Path):
        self.files = sorted(data_dir.glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No .npz files found in {data_dir}")

    def _select_files(self, max_samples=0, sample_mode="stratified", seed=0):
        files = self.files
        if max_samples <= 0 or max_samples >= len(files):
            return files

        rng = np.random.default_rng(seed)
        if sample_mode == "sequential":
            return files[:max_samples]
        if sample_mode == "random":
            idx = rng.choice(len(files), size=max_samples, replace=False)
            return [files[i] for i in sorted(idx)]

        by_class = {k: [] for k in range(10)}
        fallback = []
        for fp in files:
            k = infer_label_from_filename(fp)
            if k is None:
                fallback.append(fp)
            else:
                by_class[k].append(fp)
        if fallback:
            idx = rng.choice(len(files), size=max_samples, replace=False)
            return [files[i] for i in sorted(idx)]

        for k in range(10):
            rng.shuffle(by_class[k])

        selected = []
        base = max_samples // 10
        for k in range(10):
            take = min(base, len(by_class[k]))
            selected.extend(by_class[k][:take])
            by_class[k] = by_class[k][take:]

        cls_order = list(range(10))
        rng.shuffle(cls_order)
        while len(selected) < max_samples:
            progressed = False
            for k in cls_order:
                if by_class[k]:
                    selected.append(by_class[k].pop())
                    progressed = True
                    if len(selected) == max_samples:
                        break
            if not progressed:
                break
        return sorted(selected)

    def iter_samples(self, max_samples=0, sample_mode="stratified", seed=0, shuffle=False):
        files = self._select_files(max_samples=max_samples, sample_mode=sample_mode, seed=seed)
        idx = np.arange(len(files))
        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(idx)
        for i in idx:
            d = np.load(files[i])
            yield d["x"].astype(np.float32), int(d["y"])

    def sample_shape(self):
        d = np.load(self.files[0])
        return d["x"].shape

    def count(self, max_samples=0):
        return min(len(self.files), max_samples) if max_samples > 0 else len(self.files)


@dataclass
class MemoryStats:
    sram_weight_reads: int = 0
    sram_weight_writes: int = 0
    stdp_weight_reads: int = 0
    stdp_weight_writes: int = 0
    corr_mem_reads: int = 0
    corr_mem_writes: int = 0
    trace_reads: int = 0
    trace_writes: int = 0

    def total_fetch_reads(self):
        return self.sram_weight_reads + self.stdp_weight_reads + self.corr_mem_reads + self.trace_reads

    def total_writes(self):
        return self.sram_weight_writes + self.stdp_weight_writes + self.corr_mem_writes + self.trace_writes

    def to_dict(self):
        return {
            "sram_weight_reads": self.sram_weight_reads,
            "sram_weight_writes": self.sram_weight_writes,
            "stdp_weight_reads": self.stdp_weight_reads,
            "stdp_weight_writes": self.stdp_weight_writes,
            "corr_mem_reads": self.corr_mem_reads,
            "corr_mem_writes": self.corr_mem_writes,
            "trace_reads": self.trace_reads,
            "trace_writes": self.trace_writes,
            "total_fetch_reads": self.total_fetch_reads(),
            "total_writes": self.total_writes(),
        }

    def add(self, other):
        self.sram_weight_reads += other.sram_weight_reads
        self.sram_weight_writes += other.sram_weight_writes
        self.stdp_weight_reads += other.stdp_weight_reads
        self.stdp_weight_writes += other.stdp_weight_writes
        self.corr_mem_reads += other.corr_mem_reads
        self.corr_mem_writes += other.corr_mem_writes
        self.trace_reads += other.trace_reads
        self.trace_writes += other.trace_writes


class STDPLUT:
    def __init__(self, window=16, a_plus=0.01, a_minus=0.012):
        self.window = int(window)
        idx = np.arange(self.window + 1, dtype=np.float32)
        self.ltp_lut = float(a_plus) * (idx / max(1, self.window))
        self.ltd_lut = float(a_minus) * (idx / max(1, self.window))


class CorrelationMemory:
    def __init__(self, input_dim, hidden_dim, fan_in=16, threshold=0.9, min_steps=1000, seed=0):
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.fan_in = int(min(fan_in, input_dim))
        self.threshold = float(threshold)
        self.min_steps = int(min_steps)

        rng = np.random.default_rng(seed)
        self.fanin_idx = np.zeros((hidden_dim, self.fan_in), dtype=np.int32)
        for h in range(hidden_dim):
            self.fanin_idx[h] = rng.choice(input_dim, size=self.fan_in, replace=False)
        self.corr_counts = np.zeros((hidden_dim, self.fan_in), dtype=np.int64)
        self.total_steps = 0
        self.gate_mask = np.ones((hidden_dim,), dtype=np.uint8)
        self.correlated_input_idx = np.full((hidden_dim,), -1, dtype=np.int32)

    def update(self, pre_spk, post_spk, mem_stats: MemoryStats):
        # Reads old counters + writes new counters for each [hidden, fan_in].
        n = self.hidden_dim * self.fan_in
        mem_stats.corr_mem_reads += n
        mem_stats.corr_mem_writes += n
        pre_sel = pre_spk[self.fanin_idx]
        same = pre_sel == post_spk[:, None]
        self.corr_counts += same.astype(np.int64)
        self.total_steps += 1

    def finalize(self):
        if self.total_steps < self.min_steps:
            return
        scores = self.corr_counts.astype(np.float32) / float(self.total_steps)
        for h in range(self.hidden_dim):
            hit = np.where(scores[h] >= self.threshold)[0]
            if hit.size > 0:
                k = int(hit[0])
                self.correlated_input_idx[h] = int(self.fanin_idx[h, k])
                self.gate_mask[h] = 0


class MemoryAwareSTDPTrainer:
    def __init__(
        self,
        input_dim,
        hidden_dim=512,
        threshold=1.0,
        beta=0.9,
        leak=0.0,
        stdp_window=16,
        a_plus=0.01,
        a_minus=0.012,
        w_min=0.0,
        w_max=1.0,
        corr_fan_in=16,
        corr_threshold=0.9,
        corr_min_steps=1000,
        seed=0,
    ):
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.w_min = float(w_min)
        self.w_max = float(w_max)
        self.mem_stats = MemoryStats()

        rng = np.random.default_rng(seed)
        init_w = rng.uniform(self.w_min, self.w_max * 0.2, size=(self.input_dim, self.hidden_dim)).astype(np.float32)
        self.weight_mem = SRAMWeightMemory(rows=self.input_dim, cols=self.hidden_dim, dtype=np.float32, init="zeros")
        self.weight_mem.load_from_array(init_w)

        self.lif = LIFLayer(size=self.hidden_dim, threshold=threshold, beta=beta, leak=leak, dtype=np.float32)
        self.stdp = STDPLUT(window=stdp_window, a_plus=a_plus, a_minus=a_minus)
        self.pre_trace = np.zeros((self.input_dim,), dtype=np.int32)
        self.post_trace = np.zeros((self.hidden_dim,), dtype=np.int32)
        self.corr = CorrelationMemory(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            fan_in=corr_fan_in,
            threshold=corr_threshold,
            min_steps=corr_min_steps,
            seed=seed,
        )

    def run_sample(self, x, train=True):
        t_steps = x.shape[0]
        x_flat = x.reshape(t_steps, -1)
        self.lif.reset(batch_size=1)

        spike_count = np.zeros((self.hidden_dim,), dtype=np.float32)
        local_stats = MemoryStats()

        for t in range(t_steps):
            pre_spk = (x_flat[t] > 0).astype(np.float32)
            active_idx = np.where(pre_spk > 0)[0]

            # Event-driven SRAM read count: only active input rows fetched.
            fetch_n = int(len(active_idx) * self.hidden_dim)
            local_stats.sram_weight_reads += fetch_n
            w = self.weight_mem.export_array()
            syn_current = np.sum(w[active_idx], axis=0, dtype=np.float32) if len(active_idx) else np.zeros((self.hidden_dim,), dtype=np.float32)

            post_spk = self.lif.step(syn_current.reshape(1, -1))[0]
            spike_count += post_spk

            # Correlation counter memory update.
            self.corr.update(pre_spk, post_spk, local_stats)

            if train:
                # Trace read costs for LUT lookups.
                local_stats.trace_reads += self.input_dim + self.hidden_dim

                # STDP update touch counts.
                post_n = int(np.sum(post_spk > 0))
                pre_n = int(np.sum(pre_spk > 0))
                ltp_touched = self.input_dim * post_n
                ltd_touched = pre_n * self.hidden_dim
                touched = ltp_touched + ltd_touched
                local_stats.stdp_weight_reads += touched
                local_stats.stdp_weight_writes += touched

                if post_n > 0:
                    w += np.outer(self.stdp.ltp_lut[self.pre_trace], post_spk)
                if pre_n > 0:
                    w -= np.outer(pre_spk, self.stdp.ltd_lut[self.post_trace])
                np.clip(w, self.w_min, self.w_max, out=w)

                # SRAM write back after STDP update.
                self.weight_mem.load_from_array(w)
                local_stats.sram_weight_writes += touched

            # Trace updates.
            local_stats.trace_reads += self.input_dim + self.hidden_dim
            local_stats.trace_writes += self.input_dim + self.hidden_dim
            self.pre_trace = np.maximum(self.pre_trace - 1, 0)
            self.post_trace = np.maximum(self.post_trace - 1, 0)
            self.pre_trace[pre_spk > 0] = self.stdp.window
            self.post_trace[post_spk > 0] = self.stdp.window

        self.mem_stats.add(local_stats)
        return spike_count / max(1, t_steps), local_stats


def train_and_analyze(args):
    train_ds = NpzStreamDataset(Path(args.train_dir))
    sample_shape = train_ds.sample_shape()
    input_dim = int(np.prod(sample_shape[1:]))
    print(f"Topology: input={input_dim} -> hidden={args.hidden_dim} (STDP)")

    trainer = MemoryAwareSTDPTrainer(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        threshold=args.threshold,
        beta=args.beta,
        leak=args.leak,
        stdp_window=args.stdp_window,
        a_plus=args.a_plus,
        a_minus=args.a_minus,
        w_min=args.w_min,
        w_max=args.w_max,
        corr_fan_in=args.corr_fan_in,
        corr_threshold=args.corr_threshold,
        corr_min_steps=args.corr_min_steps,
        seed=args.seed,
    )

    epoch_reports = []
    for epoch in range(1, args.epochs + 1):
        epoch_stats = MemoryStats()
        processed = 0
        total = train_ds.count(max_samples=args.max_train)
        for x, _y in train_ds.iter_samples(
            max_samples=args.max_train,
            sample_mode=args.sample_mode,
            seed=args.seed + epoch,
            shuffle=True,
        ):
            _spk, local = trainer.run_sample(x, train=True)
            epoch_stats.add(local)
            processed += 1
            if processed % 1000 == 0 or processed == total:
                print(f"Epoch {epoch:03d} progress: {processed}/{total}")

        trainer.corr.finalize()
        gated_ratio = float(np.mean(trainer.corr.gate_mask == 0))
        report = {"epoch": epoch, "gated_ratio": gated_ratio, **epoch_stats.to_dict()}
        epoch_reports.append(report)
        print(
            f"Epoch {epoch:03d} reads={report['total_fetch_reads']:,} "
            f"writes={report['total_writes']:,} gated_ratio={gated_ratio:.4f}"
        )

    final_report = {
        "config": {
            "train_dir": args.train_dir,
            "hidden_dim": args.hidden_dim,
            "epochs": args.epochs,
            "max_train": args.max_train,
            "stdp_window": args.stdp_window,
            "a_plus": args.a_plus,
            "a_minus": args.a_minus,
            "corr_fan_in": args.corr_fan_in,
            "corr_threshold": args.corr_threshold,
            "corr_min_steps": args.corr_min_steps,
        },
        "totals": trainer.mem_stats.to_dict(),
        "epoch_reports": epoch_reports,
        "gated_ratio_final": float(np.mean(trainer.corr.gate_mask == 0)),
    }

    out_path = Path(args.report_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(final_report, indent=2), encoding="utf-8")
    print(f"Saved memory-fetch analysis report: {out_path}")

    if args.save_sram:
        sram_path = out_path.with_suffix(".weights_sram.npz")
        trainer.weight_mem.save(sram_path)
        print(f"Saved SRAM weights: {sram_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Train STDP SNN and analyze memory fetch counts.")
    p.add_argument("--train-dir", type=str, required=True)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--threshold", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=0.9)
    p.add_argument("--leak", type=float, default=0.0)
    p.add_argument("--stdp-window", type=int, default=16)
    p.add_argument("--a-plus", type=float, default=0.01)
    p.add_argument("--a-minus", type=float, default=0.012)
    p.add_argument("--w-min", type=float, default=0.0)
    p.add_argument("--w-max", type=float, default=1.0)
    p.add_argument("--corr-fan-in", type=int, default=16)
    p.add_argument("--corr-threshold", type=float, default=0.9)
    p.add_argument("--corr-min-steps", type=int, default=1000)
    p.add_argument("--sample-mode", type=str, choices=["stratified", "random", "sequential"], default="stratified")
    p.add_argument("--max-train", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--report-out", type=str, default="reports/memory_fetch_report.json")
    p.add_argument("--save-sram", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    train_and_analyze(parse_args())
