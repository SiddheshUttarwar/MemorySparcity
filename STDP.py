import argparse
import json
import re
from pathlib import Path

import numpy as np

from LIF import LIFLayer
from SRAM import SRAMWeightMemory


"""
LUT-based STDP SNN with Correlation-based Neuron Gating.

Implements:
- STDP learning via timing traces + LUT updates.
- Correlation counter memory (Hamming similarity style) between each post neuron
  and a sampled set of presynaptic fan-ins.
- Threshold-based correlated-source detection.
- Post-training gating + rewiring: gated neuron output is replaced by correlated
  presynaptic spike train to reduce redundant neuron activity.
"""


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
            x = d["x"].astype(np.float32)
            y = int(d["y"])
            yield x, y

    def sample_shape(self):
        d = np.load(self.files[0])
        return d["x"].shape

    def count(self, max_samples=0):
        return min(len(self.files), max_samples) if max_samples > 0 else len(self.files)


class STDPLUT:
    def __init__(self, window=16, a_plus=0.01, a_minus=0.012):
        self.window = int(window)
        idx = np.arange(self.window + 1, dtype=np.float32)
        self.ltp_lut = float(a_plus) * (idx / max(1, self.window))
        self.ltd_lut = float(a_minus) * (idx / max(1, self.window))

    def ltp_values(self, pre_trace):
        return self.ltp_lut[pre_trace]

    def ltd_values(self, post_trace):
        return self.ltd_lut[post_trace]


class CorrelationGater:
    """
    Correlation counter memory + gating decision.

    For each post neuron, sample `fan_in` presynaptic inputs and accumulate
    similarity count where pre bit equals post bit (Hamming similarity).
    """

    def __init__(self, input_dim, hidden_dim, fan_in=16, corr_threshold=0.9, min_steps=1000, seed=0):
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.fan_in = int(min(fan_in, input_dim))
        self.corr_threshold = float(corr_threshold)
        self.min_steps = int(min_steps)

        rng = np.random.default_rng(seed)
        self.fanin_idx = np.zeros((self.hidden_dim, self.fan_in), dtype=np.int32)
        for h in range(self.hidden_dim):
            self.fanin_idx[h] = rng.choice(self.input_dim, size=self.fan_in, replace=False)

        self.corr_counts = np.zeros((self.hidden_dim, self.fan_in), dtype=np.int64)
        self.total_steps = 0

        self.correlated_input_idx = np.full((self.hidden_dim,), -1, dtype=np.int32)
        self.gate_mask = np.ones((self.hidden_dim,), dtype=np.uint8)  # 1=active, 0=gated

    def update(self, pre_spk, post_spk):
        # pre_spk: [input_dim], post_spk: [hidden_dim] (binary)
        pre_sel = pre_spk[self.fanin_idx]  # [hidden, fan_in]
        post_col = post_spk[:, None]  # [hidden, 1]
        same = (pre_sel == post_col)
        self.corr_counts += same.astype(np.int64)
        self.total_steps += 1

    def finalize(self):
        if self.total_steps < self.min_steps:
            return

        scores = self.corr_counts.astype(np.float32) / float(self.total_steps)
        for h in range(self.hidden_dim):
            hit = np.where(scores[h] >= self.corr_threshold)[0]
            if hit.size > 0:
                k = int(hit[0])  # first correlated fan-in, consistent with architecture note
                self.correlated_input_idx[h] = int(self.fanin_idx[h, k])
                self.gate_mask[h] = 0

    def apply_rewire(self, post_spk, pre_spk):
        # For gated neurons, replace output with correlated input spike.
        out = post_spk.copy()
        gated = np.where(self.gate_mask == 0)[0]
        if gated.size > 0:
            src = self.correlated_input_idx[gated]
            valid = src >= 0
            if np.any(valid):
                g = gated[valid]
                s = src[valid]
                out[g] = pre_spk[s]
        return out

    def gated_ratio(self):
        return float(np.mean(self.gate_mask == 0))


class STDPSNN:
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

        rng = np.random.default_rng(seed)
        self.w = rng.uniform(self.w_min, self.w_max * 0.2, size=(self.input_dim, self.hidden_dim)).astype(np.float32)

        self.lif = LIFLayer(
            size=self.hidden_dim,
            threshold=threshold,
            beta=beta,
            leak=leak,
            v_rest=0.0,
            v_reset=0.0,
            dtype=np.float32,
        )
        self.stdp = STDPLUT(window=stdp_window, a_plus=a_plus, a_minus=a_minus)
        self.gater = CorrelationGater(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            fan_in=corr_fan_in,
            corr_threshold=corr_threshold,
            min_steps=corr_min_steps,
            seed=seed,
        )

    def run_sample(self, x, train=False, use_gating=False):
        t_steps = x.shape[0]
        x_flat = x.reshape(t_steps, -1)

        self.lif.reset(batch_size=1)
        pre_trace = np.zeros((self.input_dim,), dtype=np.int32)
        post_trace = np.zeros((self.hidden_dim,), dtype=np.int32)
        spike_count = np.zeros((self.hidden_dim,), dtype=np.float32)

        for t in range(t_steps):
            pre_spk = (x_flat[t] > 0).astype(np.float32)

            current = pre_spk @ self.w
            post_spk_raw = self.lif.step(current.reshape(1, -1))[0]

            if train:
                self.gater.update(pre_spk, post_spk_raw)

                if np.any(post_spk_raw > 0):
                    ltp_vec = self.stdp.ltp_values(pre_trace)
                    self.w += np.outer(ltp_vec, post_spk_raw)

                if np.any(pre_spk > 0):
                    ltd_vec = self.stdp.ltd_values(post_trace)
                    self.w -= np.outer(pre_spk, ltd_vec)

                np.clip(self.w, self.w_min, self.w_max, out=self.w)

            post_spk = self.gater.apply_rewire(post_spk_raw, pre_spk) if use_gating else post_spk_raw
            spike_count += post_spk

            pre_trace = np.maximum(pre_trace - 1, 0)
            post_trace = np.maximum(post_trace - 1, 0)
            pre_trace[pre_spk > 0] = self.stdp.window
            post_trace[post_spk_raw > 0] = self.stdp.window

        return spike_count / max(1, t_steps)


def assign_neuron_labels(spike_sum_by_label):
    best = np.argmax(spike_sum_by_label, axis=0)
    active = np.sum(spike_sum_by_label, axis=0) > 0
    labels = np.full((spike_sum_by_label.shape[1],), -1, dtype=np.int64)
    labels[active] = best[active]
    return labels


def predict_from_spikes(spike_rate, neuron_labels):
    votes = np.zeros((10,), dtype=np.float32)
    for n, cls in enumerate(neuron_labels):
        if cls >= 0:
            votes[cls] += spike_rate[n]
    return int(np.argmax(votes))


def evaluate(model, dataset, neuron_labels, max_samples=0, sample_mode="stratified", seed=0, use_gating=False):
    correct = 0
    total = 0
    for x, y in dataset.iter_samples(max_samples=max_samples, sample_mode=sample_mode, seed=seed, shuffle=False):
        spk = model.run_sample(x, train=False, use_gating=use_gating)
        pred = predict_from_spikes(spk, neuron_labels)
        correct += int(pred == y)
        total += 1
    return (correct / total) if total > 0 else 0.0


def train(args):
    train_ds = NpzStreamDataset(Path(args.train_dir))
    test_ds = NpzStreamDataset(Path(args.test_dir))

    sample_shape = train_ds.sample_shape()
    if len(sample_shape) != 4:
        raise ValueError(f"Expected sample shape [T,2,H,W], got {sample_shape}")

    input_dim = int(np.prod(sample_shape[1:]))
    print(f"STDP SNN topology: input={input_dim} -> hidden={args.hidden_dim} -> output=10")

    model = STDPSNN(
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

        # Build/update gating decision from accumulated correlation memory.
        model.gater.finalize()
        neuron_labels = assign_neuron_labels(spike_sum_by_label)

        train_acc = evaluate(
            model,
            train_ds,
            neuron_labels,
            max_samples=min(args.eval_train_samples, total),
            sample_mode=args.sample_mode,
            seed=args.seed,
            use_gating=args.enable_gating,
        )
        test_acc = evaluate(
            model,
            test_ds,
            neuron_labels,
            max_samples=args.max_test,
            sample_mode=args.sample_mode,
            seed=args.seed,
            use_gating=args.enable_gating,
        )
        print(
            f"Epoch {epoch:03d} | train_acc={train_acc:.4f} | test_acc={test_acc:.4f} | "
            f"gated_ratio={model.gater.gated_ratio():.4f}"
        )

    neuron_labels = assign_neuron_labels(spike_sum_by_label)
    final_train_acc = evaluate(
        model,
        train_ds,
        neuron_labels,
        max_samples=min(args.eval_train_samples, train_ds.count(max_samples=args.max_train)),
        sample_mode=args.sample_mode,
        seed=args.seed,
        use_gating=args.enable_gating,
    )
    final_test_acc = evaluate(
        model,
        test_ds,
        neuron_labels,
        max_samples=args.max_test,
        sample_mode=args.sample_mode,
        seed=args.seed,
        use_gating=args.enable_gating,
    )
    print(f"Final train accuracy: {final_train_acc:.4f}")
    print(f"Final test accuracy:  {final_test_acc:.4f}")

    model_path = Path(args.model_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        model_path,
        w_in=model.w,
        neuron_labels=neuron_labels,
        hidden_dim=np.int64(args.hidden_dim),
        threshold=np.float32(args.threshold),
        beta=np.float32(args.beta),
        leak=np.float32(args.leak),
        stdp_window=np.int64(args.stdp_window),
        a_plus=np.float32(args.a_plus),
        a_minus=np.float32(args.a_minus),
        w_min=np.float32(args.w_min),
        w_max=np.float32(args.w_max),
        corr_fanin_idx=model.gater.fanin_idx,
        corr_counts=model.gater.corr_counts,
        corr_total_steps=np.int64(model.gater.total_steps),
        corr_threshold=np.float32(model.gater.corr_threshold),
        corr_min_steps=np.int64(model.gater.min_steps),
        corr_input_idx=model.gater.correlated_input_idx,
        gate_mask=model.gater.gate_mask,
        enable_gating=np.int64(1 if args.enable_gating else 0),
    )
    print(f"Saved STDP model: {model_path}")

    sram_dir = model_path.parent / f"{model_path.stem}_sram"
    sram_dir.mkdir(parents=True, exist_ok=True)
    sram_in = SRAMWeightMemory(rows=model.w.shape[0], cols=model.w.shape[1], init="zeros")
    sram_in.load_from_array(model.w)
    sram_in.save(sram_dir / "W_in_stdp.npz")
    print(f"Saved SRAM weight memory: {sram_dir / 'W_in_stdp.npz'}")

    metrics = {
        "train_acc": final_train_acc,
        "test_acc": final_test_acc,
        "hidden_dim": args.hidden_dim,
        "stdp_window": args.stdp_window,
        "a_plus": args.a_plus,
        "a_minus": args.a_minus,
        "w_min": args.w_min,
        "w_max": args.w_max,
        "corr_fan_in": args.corr_fan_in,
        "corr_threshold": args.corr_threshold,
        "corr_min_steps": args.corr_min_steps,
        "gated_ratio": model.gater.gated_ratio(),
        "enable_gating": bool(args.enable_gating),
    }
    metrics_path = model_path.with_suffix(".json")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved metrics: {metrics_path}")


def predict(args):
    m = np.load(Path(args.model_path), allow_pickle=True)
    w = m["w_in"].astype(np.float32)
    neuron_labels = m["neuron_labels"].astype(np.int64)

    ds = NpzStreamDataset(Path(args.data_dir))
    sample_shape = ds.sample_shape()
    input_dim = int(np.prod(sample_shape[1:]))
    if input_dim != w.shape[0]:
        raise ValueError(f"Input mismatch: model input={w.shape[0]}, data input={input_dim}")

    model = STDPSNN(
        input_dim=w.shape[0],
        hidden_dim=w.shape[1],
        threshold=float(m["threshold"]),
        beta=float(m["beta"]),
        leak=float(m["leak"]),
        stdp_window=int(m["stdp_window"]),
        a_plus=float(m["a_plus"]),
        a_minus=float(m["a_minus"]),
        w_min=float(m["w_min"]),
        w_max=float(m["w_max"]),
        corr_fan_in=int(m["corr_fanin_idx"].shape[1]),
        corr_threshold=float(m["corr_threshold"]),
        corr_min_steps=int(m["corr_min_steps"]),
        seed=0,
    )
    model.w = w
    model.gater.fanin_idx = m["corr_fanin_idx"].astype(np.int32)
    model.gater.corr_counts = m["corr_counts"].astype(np.int64)
    model.gater.total_steps = int(m["corr_total_steps"])
    model.gater.correlated_input_idx = m["corr_input_idx"].astype(np.int32)
    model.gater.gate_mask = m["gate_mask"].astype(np.uint8)

    use_gating = args.enable_gating
    acc = evaluate(
        model,
        ds,
        neuron_labels,
        max_samples=args.max_samples,
        sample_mode=args.sample_mode,
        seed=args.seed,
        use_gating=use_gating,
    )
    total = ds.count(max_samples=args.max_samples)
    print(f"Prediction accuracy: {acc:.4f} on {total} samples | gated_ratio={model.gater.gated_ratio():.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="LUT-based STDP SNN with correlation-based neuron gating")
    sub = parser.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train", help="Train with LUT-based STDP")
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
    tr.add_argument("--model-out", type=str, default="checkpoints/snn_stdp.npz")

    pr = sub.add_parser("predict", help="Predict using saved STDP model")
    pr.add_argument("--model-path", type=str, required=True)
    pr.add_argument("--data-dir", type=str, required=True)
    pr.add_argument("--enable-gating", action="store_true")
    pr.add_argument("--sample-mode", type=str, choices=["stratified", "random", "sequential"], default="stratified")
    pr.add_argument("--max-samples", type=int, default=0)
    pr.add_argument("--seed", type=int, default=0)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.cmd == "train":
        train(args)
    elif args.cmd == "predict":
        predict(args)
