import argparse
import json
import re
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


class SpikeRouter:
    def route(self, x_t):
        # x_t: [2, H, W] -> flattened pre spikes.
        return (x_t.reshape(-1) > 0).astype(np.float32)


class SynapticWeightMemory:
    def __init__(self, input_dim, hidden_dim, w_min=0.0, w_max=1.0, seed=0):
        self.w_min = float(w_min)
        self.w_max = float(w_max)
        rng = np.random.default_rng(seed)
        init_w = rng.uniform(self.w_min, self.w_max * 0.2, size=(input_dim, hidden_dim)).astype(np.float32)
        self.mem = SRAMWeightMemory(rows=input_dim, cols=hidden_dim, dtype=np.float32, init="zeros")
        self.mem.load_from_array(init_w)

    def read_all(self):
        return self.mem.export_array()

    def write_all(self, w):
        np.clip(w, self.w_min, self.w_max, out=w)
        self.mem.load_from_array(w)

    def save(self, path):
        self.mem.save(path)


class SynapseArray:
    def __init__(self, weight_mem: SynapticWeightMemory):
        self.weight_mem = weight_mem
        self.synapse_gate_mask = None  # optional [input_dim, hidden_dim], 1=enabled

    def set_synapse_gate_mask(self, mask):
        self.synapse_gate_mask = mask.astype(np.uint8)

    def accumulate(self, pre_spikes):
        w = self.weight_mem.read_all()
        if self.synapse_gate_mask is not None:
            w = w * self.synapse_gate_mask
        return pre_spikes @ w


class PrePostSpikeTap:
    def __init__(self, input_dim, hidden_dim, window=16):
        self.window = int(window)
        self.pre_trace = np.zeros((input_dim,), dtype=np.int32)
        self.post_trace = np.zeros((hidden_dim,), dtype=np.int32)

    def update(self, pre_spk, post_spk):
        self.pre_trace = np.maximum(self.pre_trace - 1, 0)
        self.post_trace = np.maximum(self.post_trace - 1, 0)
        self.pre_trace[pre_spk > 0] = self.window
        self.post_trace[post_spk > 0] = self.window


class LUTSTDPUnit:
    def __init__(self, window=16, a_plus=0.01, a_minus=0.012):
        self.window = int(window)
        idx = np.arange(self.window + 1, dtype=np.float32)
        self.ltp_lut = float(a_plus) * (idx / max(1, self.window))
        self.ltd_lut = float(a_minus) * (idx / max(1, self.window))

    def update_weights(self, w, pre_trace, post_trace, pre_spk, post_spk):
        if np.any(post_spk > 0):
            w += np.outer(self.ltp_lut[pre_trace], post_spk)
        if np.any(pre_spk > 0):
            w -= np.outer(pre_spk, self.ltd_lut[post_trace])


class CorrelationBasedNeuronGating:
    def __init__(self, input_dim, hidden_dim, fan_in=16, corr_threshold=0.9, min_steps=1000, seed=0):
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.fan_in = int(min(fan_in, input_dim))
        self.corr_threshold = float(corr_threshold)
        self.min_steps = int(min_steps)

        rng = np.random.default_rng(seed)
        self.fanin_idx = np.zeros((hidden_dim, self.fan_in), dtype=np.int32)
        for h in range(hidden_dim):
            self.fanin_idx[h] = rng.choice(input_dim, size=self.fan_in, replace=False)

        self.corr_counts = np.zeros((hidden_dim, self.fan_in), dtype=np.int64)
        self.total_steps = 0
        self.correlated_input_idx = np.full((hidden_dim,), -1, dtype=np.int32)
        self.neuron_gate_mask = np.ones((hidden_dim,), dtype=np.uint8)

    def update(self, pre_spk, post_spk):
        pre_sel = pre_spk[self.fanin_idx]
        same = pre_sel == post_spk[:, None]
        self.corr_counts += same.astype(np.int64)
        self.total_steps += 1

    def finalize(self):
        if self.total_steps < self.min_steps:
            return
        scores = self.corr_counts.astype(np.float32) / float(self.total_steps)
        for h in range(self.hidden_dim):
            hit = np.where(scores[h] >= self.corr_threshold)[0]
            if hit.size > 0:
                i = int(hit[0])
                self.correlated_input_idx[h] = int(self.fanin_idx[h, i])
                self.neuron_gate_mask[h] = 0

    def apply_rewire(self, post_spk, pre_spk):
        out = post_spk.copy()
        gated = np.where(self.neuron_gate_mask == 0)[0]
        if gated.size > 0:
            src = self.correlated_input_idx[gated]
            valid = src >= 0
            if np.any(valid):
                out[gated[valid]] = pre_spk[src[valid]]
        return out

    def gated_ratio(self):
        return float(np.mean(self.neuron_gate_mask == 0))


class ReadoutClassifier:
    @staticmethod
    def assign_neuron_labels(spike_sum_by_label):
        best = np.argmax(spike_sum_by_label, axis=0)
        active = np.sum(spike_sum_by_label, axis=0) > 0
        labels = np.full((spike_sum_by_label.shape[1],), -1, dtype=np.int64)
        labels[active] = best[active]
        return labels

    @staticmethod
    def predict(spike_rate, neuron_labels):
        votes = np.zeros((10,), dtype=np.float32)
        for n, cls in enumerate(neuron_labels):
            if cls >= 0:
                votes[cls] += spike_rate[n]
        return int(np.argmax(votes))


class NeuromorphicArchitecture:
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
        self.router = SpikeRouter()
        self.weight_mem = SynapticWeightMemory(input_dim, hidden_dim, w_min=w_min, w_max=w_max, seed=seed)
        self.synapse_array = SynapseArray(self.weight_mem)
        self.lif = LIFLayer(size=hidden_dim, threshold=threshold, beta=beta, leak=leak, dtype=np.float32)
        self.spike_taps = PrePostSpikeTap(input_dim=input_dim, hidden_dim=hidden_dim, window=stdp_window)
        self.stdp = LUTSTDPUnit(window=stdp_window, a_plus=a_plus, a_minus=a_minus)
        self.gating = CorrelationBasedNeuronGating(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            fan_in=corr_fan_in,
            corr_threshold=corr_threshold,
            min_steps=corr_min_steps,
            seed=seed,
        )
        self.hidden_dim = hidden_dim
        self.w_min = w_min
        self.w_max = w_max

    def run_sample(self, x, train=False, use_gating=False):
        t_steps = x.shape[0]
        self.lif.reset(batch_size=1)
        spike_count = np.zeros((self.hidden_dim,), dtype=np.float32)

        for t in range(t_steps):
            pre_spk = self.router.route(x[t])
            syn_current = self.synapse_array.accumulate(pre_spk)
            post_spk_raw = self.lif.step(syn_current.reshape(1, -1))[0]

            if train:
                self.gating.update(pre_spk, post_spk_raw)
                w = self.weight_mem.read_all()
                self.stdp.update_weights(
                    w=w,
                    pre_trace=self.spike_taps.pre_trace,
                    post_trace=self.spike_taps.post_trace,
                    pre_spk=pre_spk,
                    post_spk=post_spk_raw,
                )
                self.weight_mem.write_all(w)

            post_spk = self.gating.apply_rewire(post_spk_raw, pre_spk) if use_gating else post_spk_raw
            spike_count += post_spk
            self.spike_taps.update(pre_spk, post_spk_raw)

        return spike_count / max(1, t_steps)


def evaluate(model, dataset, neuron_labels, max_samples=0, sample_mode="stratified", seed=0, use_gating=False):
    correct = 0
    total = 0
    for x, y in dataset.iter_samples(max_samples=max_samples, sample_mode=sample_mode, seed=seed, shuffle=False):
        spk = model.run_sample(x, train=False, use_gating=use_gating)
        pred = ReadoutClassifier.predict(spk, neuron_labels)
        correct += int(pred == y)
        total += 1
    return (correct / total) if total > 0 else 0.0


def train(args):
    train_ds = NpzStreamDataset(Path(args.train_dir))
    test_ds = NpzStreamDataset(Path(args.test_dir))

    sample_shape = train_ds.sample_shape()
    input_dim = int(np.prod(sample_shape[1:]))
    print(f"Architecture topology: input={input_dim} -> hidden={args.hidden_dim} -> output=10")

    model = NeuromorphicArchitecture(
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

    spike_sum_by_label = np.zeros((10, args.hidden_dim), dtype=np.float64)
    for epoch in range(1, args.epochs + 1):
        processed = 0
        total = train_ds.count(max_samples=args.max_train)
        for x, y in train_ds.iter_samples(
            max_samples=args.max_train,
            sample_mode=args.sample_mode,
            seed=args.seed + epoch,
            shuffle=True,
        ):
            spk = model.run_sample(x, train=True, use_gating=False)
            spike_sum_by_label[y] += spk
            processed += 1
            if processed % 2000 == 0 or processed == total:
                print(f"Epoch {epoch:03d} progress: {processed}/{total}")

        model.gating.finalize()
        neuron_labels = ReadoutClassifier.assign_neuron_labels(spike_sum_by_label)
        tr_acc = evaluate(
            model,
            train_ds,
            neuron_labels,
            max_samples=min(args.eval_train_samples, total),
            sample_mode=args.sample_mode,
            seed=args.seed,
            use_gating=args.enable_gating,
        )
        te_acc = evaluate(
            model,
            test_ds,
            neuron_labels,
            max_samples=args.max_test,
            sample_mode=args.sample_mode,
            seed=args.seed,
            use_gating=args.enable_gating,
        )
        print(f"Epoch {epoch:03d} | train_acc={tr_acc:.4f} | test_acc={te_acc:.4f} | gated={model.gating.gated_ratio():.4f}")

    neuron_labels = ReadoutClassifier.assign_neuron_labels(spike_sum_by_label)
    final_train = evaluate(
        model,
        train_ds,
        neuron_labels,
        max_samples=min(args.eval_train_samples, train_ds.count(max_samples=args.max_train)),
        sample_mode=args.sample_mode,
        seed=args.seed,
        use_gating=args.enable_gating,
    )
    final_test = evaluate(
        model,
        test_ds,
        neuron_labels,
        max_samples=args.max_test,
        sample_mode=args.sample_mode,
        seed=args.seed,
        use_gating=args.enable_gating,
    )
    print(f"Final train accuracy: {final_train:.4f}")
    print(f"Final test accuracy:  {final_test:.4f}")

    model_path = Path(args.model_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        model_path,
        w_in=model.weight_mem.read_all(),
        neuron_labels=neuron_labels,
        gate_mask=model.gating.neuron_gate_mask,
        corr_input_idx=model.gating.correlated_input_idx,
        corr_fanin_idx=model.gating.fanin_idx,
        corr_counts=model.gating.corr_counts,
        corr_total_steps=np.int64(model.gating.total_steps),
        enable_gating=np.int64(1 if args.enable_gating else 0),
    )
    sram_path = model_path.parent / f"{model_path.stem}_W_in_sram.npz"
    model.weight_mem.save(sram_path)

    metrics = {
        "train_acc": final_train,
        "test_acc": final_test,
        "gated_ratio": model.gating.gated_ratio(),
        "hidden_dim": args.hidden_dim,
    }
    model_path.with_suffix(".json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved model: {model_path}")
    print(f"Saved SRAM weights: {sram_path}")


def predict(args):
    m = np.load(Path(args.model_path), allow_pickle=True)
    w = m["w_in"].astype(np.float32)
    labels = m["neuron_labels"].astype(np.int64)

    ds = NpzStreamDataset(Path(args.data_dir))
    input_dim = int(np.prod(ds.sample_shape()[1:]))
    if input_dim != w.shape[0]:
        raise ValueError(f"Input mismatch: model={w.shape[0]}, data={input_dim}")

    model = NeuromorphicArchitecture(
        input_dim=w.shape[0],
        hidden_dim=w.shape[1],
        seed=0,
    )
    model.weight_mem.write_all(w)
    if "gate_mask" in m:
        model.gating.neuron_gate_mask = m["gate_mask"].astype(np.uint8)
    if "corr_input_idx" in m:
        model.gating.correlated_input_idx = m["corr_input_idx"].astype(np.int32)

    acc = evaluate(
        model,
        ds,
        labels,
        max_samples=args.max_samples,
        sample_mode=args.sample_mode,
        seed=args.seed,
        use_gating=args.enable_gating,
    )
    total = ds.count(max_samples=args.max_samples)
    print(f"Prediction accuracy: {acc:.4f} on {total} samples")


def parse_args():
    p = argparse.ArgumentParser(description="Full neuromorphic architecture: router + synapse SRAM + LIF + STDP + correlation gating")
    sub = p.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train")
    tr.add_argument("--train-dir", type=str, required=True)
    tr.add_argument("--test-dir", type=str, required=True)
    tr.add_argument("--epochs", type=int, default=2)
    tr.add_argument("--hidden-dim", type=int, default=512)
    tr.add_argument("--threshold", type=float, default=1.0)
    tr.add_argument("--beta", type=float, default=0.9)
    tr.add_argument("--leak", type=float, default=0.0)
    tr.add_argument("--stdp-window", type=int, default=16)
    tr.add_argument("--a-plus", type=float, default=0.01)
    tr.add_argument("--a-minus", type=float, default=0.012)
    tr.add_argument("--w-min", type=float, default=0.0)
    tr.add_argument("--w-max", type=float, default=1.0)
    tr.add_argument("--corr-fan-in", type=int, default=16)
    tr.add_argument("--corr-threshold", type=float, default=0.9)
    tr.add_argument("--corr-min-steps", type=int, default=1000)
    tr.add_argument("--enable-gating", action="store_true")
    tr.add_argument("--sample-mode", type=str, choices=["stratified", "random", "sequential"], default="stratified")
    tr.add_argument("--max-train", type=int, default=0)
    tr.add_argument("--max-test", type=int, default=0)
    tr.add_argument("--eval-train-samples", type=int, default=2000)
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--model-out", type=str, default="checkpoints/full_arch_snn.npz")

    pr = sub.add_parser("predict")
    pr.add_argument("--model-path", type=str, required=True)
    pr.add_argument("--data-dir", type=str, required=True)
    pr.add_argument("--enable-gating", action="store_true")
    pr.add_argument("--sample-mode", type=str, choices=["stratified", "random", "sequential"], default="stratified")
    pr.add_argument("--max-samples", type=int, default=0)
    pr.add_argument("--seed", type=int, default=0)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.cmd == "train":
        train(args)
    elif args.cmd == "predict":
        predict(args)
