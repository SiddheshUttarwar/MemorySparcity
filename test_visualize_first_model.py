import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def softmax(logits):
    z = logits - np.max(logits, axis=1, keepdims=True)
    e = np.exp(z)
    return e / (np.sum(e, axis=1, keepdims=True) + 1e-12)


def load_dataset(data_dir: Path, max_samples: int = 0):
    files = sorted(data_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found in {data_dir}")
    if max_samples > 0:
        files = files[:max_samples]

    xs, ys, names = [], [], []
    for fp in files:
        d = np.load(fp)
        xs.append(d["x"].astype(np.float32))
        ys.append(int(d["y"]))
        names.append(fp.name)
    return np.stack(xs, axis=0), np.array(ys, dtype=np.int64), names


def extract_features_numpy_model(x, w_in, beta, threshold):
    # x: [N,T,2,H,W], w_in: [input_dim, hidden_dim]
    n, t = x.shape[0], x.shape[1]
    x_flat = x.reshape(n, t, -1)
    hidden = w_in.shape[1]

    mem = np.zeros((n, hidden), dtype=np.float32)
    spk_counts = np.zeros((n, hidden), dtype=np.float32)

    for ti in range(t):
        cur = x_flat[:, ti, :] @ w_in
        mem = beta * mem + cur
        spk = (mem >= threshold).astype(np.float32)
        mem = mem * (1.0 - spk)
        spk_counts += spk

    return spk_counts / max(1, t)


def confusion_matrix(y_true, y_pred, num_classes=10):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for yt, yp in zip(y_true, y_pred):
        cm[int(yt), int(yp)] += 1
    return cm


def class_accuracy(cm):
    acc = np.zeros((cm.shape[0],), dtype=np.float32)
    for k in range(cm.shape[0]):
        denom = np.sum(cm[k])
        acc[k] = (cm[k, k] / denom) if denom > 0 else 0.0
    return acc


def make_sample_image(x_sample):
    # x_sample: [T,2,H,W]
    on = x_sample[:, 1].sum(axis=0)
    off = x_sample[:, 0].sum(axis=0)
    return on - off


def main():
    parser = argparse.ArgumentParser(description="Test and visualize first trained NumPy SNN model.")
    parser.add_argument("--model-path", type=Path, default=Path("checkpoints/snn_numpy_smoke.npz"))
    parser.add_argument("--data-dir", type=Path, default=Path("preprocessed_test"))
    parser.add_argument("--max-samples", type=int, default=2000)
    parser.add_argument("--num-display", type=int, default=8)
    parser.add_argument("--save-fig", type=Path, default=Path("reports/first_model_test_visualization.png"))
    parser.add_argument("--save-report", type=Path, default=Path("reports/first_model_test_report.json"))
    args = parser.parse_args()

    m = np.load(args.model_path, allow_pickle=True)
    required = ["w_in", "beta", "threshold", "w_out", "b_out"]
    missing = [k for k in required if k not in m]
    if missing:
        raise ValueError(f"Model file missing keys {missing}. This script is for snn_numpy-style model.")

    w_in = m["w_in"].astype(np.float32)
    beta = float(m["beta"])
    threshold = float(m["threshold"])
    w_out = m["w_out"].astype(np.float32)
    b_out = m["b_out"].astype(np.float32)

    x, y_true, names = load_dataset(args.data_dir, max_samples=args.max_samples)
    feats = extract_features_numpy_model(x, w_in, beta, threshold)
    logits = feats @ w_out + b_out
    probs = softmax(logits)
    y_pred = np.argmax(probs, axis=1)
    conf = np.max(probs, axis=1)

    acc = float(np.mean(y_pred == y_true)) if len(y_true) else 0.0
    cm = confusion_matrix(y_true, y_pred, num_classes=10)
    cls_acc = class_accuracy(cm)

    print("=== First Model Test Summary ===")
    print(f"Model: {args.model_path}")
    print(f"Data: {args.data_dir}")
    print(f"Samples evaluated: {len(y_true)}")
    print(f"Accuracy: {acc:.4f}")

    report = {
        "model_path": str(args.model_path),
        "data_dir": str(args.data_dir),
        "num_samples": int(len(y_true)),
        "accuracy": acc,
        "confusion_matrix": cm.tolist(),
        "class_accuracy": cls_acc.tolist(),
        "mean_confidence": float(np.mean(conf)) if len(conf) else 0.0,
    }
    args.save_report.parent.mkdir(parents=True, exist_ok=True)
    args.save_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved report: {args.save_report}")

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.0, 1.0, 1.2])

    # Confusion matrix
    ax0 = fig.add_subplot(gs[0, 0:2])
    im = ax0.imshow(cm, cmap="Blues")
    ax0.set_title("Confusion Matrix")
    ax0.set_xlabel("Predicted")
    ax0.set_ylabel("True")
    ax0.set_xticks(np.arange(10))
    ax0.set_yticks(np.arange(10))
    fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)

    # Class accuracy
    ax1 = fig.add_subplot(gs[0, 2])
    ax1.bar(np.arange(10), cls_acc, color="tab:green")
    ax1.set_ylim(0, 1.0)
    ax1.set_title("Class Accuracy")
    ax1.set_xlabel("Digit")
    ax1.set_ylabel("Accuracy")

    # Confidence histogram
    ax2 = fig.add_subplot(gs[0, 3])
    ax2.hist(conf, bins=20, color="tab:orange", edgecolor="black", alpha=0.8)
    ax2.set_title("Prediction Confidence")
    ax2.set_xlabel("Max softmax probability")
    ax2.set_ylabel("Count")

    # Sample predictions
    k = min(args.num_display, len(y_true), 8)
    for i in range(k):
        ax = fig.add_subplot(gs[1 + i // 4, i % 4])
        img = make_sample_image(x[i])
        vmax = np.max(np.abs(img)) + 1e-6
        ax.imshow(img, cmap="seismic", vmin=-vmax, vmax=vmax)
        ok = "T" if y_true[i] == y_pred[i] else "F"
        ax.set_title(f"{names[i]}\ntrue={y_true[i]} pred={y_pred[i]} ({ok})")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f"First Model Test Visualization | Acc={acc:.4f}", fontsize=14)
    fig.tight_layout()

    args.save_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.save_fig, dpi=160, bbox_inches="tight")
    print(f"Saved figure: {args.save_fig}")

    plt.show()


if __name__ == "__main__":
    main()
