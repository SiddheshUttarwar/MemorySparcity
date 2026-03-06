import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from snn_model import LeNet5_CSNN

class NMNISTDataset(Dataset):
    """
    Loads preprocessed pure spatiotemporal spike tensors from disk.
    Expected data structure: [Time, Channels, Height, Width]
    """
    def __init__(self, data_dir, split='train', max_samples=None):
        pattern = os.path.join(data_dir, f'{split}_*.npz')
        self.files = glob.glob(pattern)

        if max_samples is not None and len(self.files) > max_samples:
            self.files = self.files[:max_samples]

        if len(self.files) == 0:
            raise FileNotFoundError(
                f"No files found matching '{pattern}'.\n"
                f"Make sure preprocess_dataset.py ran successfully and saved files to '{data_dir}'."
            )

        # Validate keys in the first file to catch preprocessing mismatches early
        sample = np.load(self.files[0])
        if 'data' not in sample or 'digit' not in sample:
            raise KeyError(
                f"Expected keys 'data' and 'digit' not found in {self.files[0]}.\n"
                f"Found keys: {list(sample.keys())}.\n"
                f"Delete your preprocessed_data_native folder and re-run preprocess_dataset.py."
            )

        print(f"[Dataset] Loaded {len(self.files)} samples for '{split}' split.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        # Shape: [T=20, C=2, H=28, W=28], stored as bool -> cast to float
        spike_tensor = torch.from_numpy(data['data']).float()
        label = int(data['digit'])
        return spike_tensor, label


def train_snn(epochs=20, lr=1e-3, batch_size=32, data_dir='preprocessed_data_native'):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing STBP Training on Device: {DEVICE}")

    # ── Dataset & DataLoaders ────────────────────────────────────────────────
    train_dataset = NMNISTDataset(data_dir=data_dir, split='train')
    test_dataset  = NMNISTDataset(data_dir=data_dir, split='test')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # ── Model ────────────────────────────────────────────────────────────────
    # The LeNet5_CSNN uses Soft Reset LIF with beta=0.9, V_th=1.0.
    # Surrogate gradient alpha is set to 10.0 in snn_model.py for sharp gradients.
    model = LeNet5_CSNN(in_channels=2, num_classes=10).to(DEVICE)

    # ── Optimiser & LR Scheduler ─────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # CosineAnnealingLR decays LR smoothly, greatly helping SNN convergence
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Cross-Entropy on spike rates: class with highest mean firing rate wins
    criterion = nn.CrossEntropyLoss()

    print(f"\n--- Starting CSNN Training (STBP) | {epochs} epochs | lr={lr} ---")

    best_val_acc = 0.0

    for epoch in range(epochs):
        # ── Training ──────────────────────────────────────────────────────────
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()

            # NOTE: sync_from_sram() is intentionally NOT called per-batch.
            # Doing so would overwrite BatchNorm running statistics, destroying
            # convergence. PyTorch's parameter graph is the ground truth during
            # training; SRAM is updated as a mirror at the end of each epoch.
            spike_rate, _ = model(inputs)

            loss = criterion(spike_rate, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = spike_rate.max(1)
            total   += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        scheduler.step()
        train_acc  = 100.0 * correct / total
        train_loss = total_loss / len(train_loader)

        # Mirror updated weights into SRAM once per epoch (not per batch)
        model.sync_to_sram()

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        # For validation we load from SRAM to verify the stored copy is correct
        model.sync_from_sram()

        test_loss, test_correct, test_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                spike_rate, _ = model(inputs)
                loss = criterion(spike_rate, targets)
                test_loss    += loss.item()
                _, predicted  = spike_rate.max(1)
                test_total   += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()

        val_acc  = 100.0 * test_correct / test_total
        val_loss = test_loss / len(test_loader)
        lr_now   = scheduler.get_last_lr()[0]

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_tag = " [*best]"
        else:
            best_tag = ""

        print(
            f"Epoch [{epoch+1:>2}/{epochs}] "
            f"lr={lr_now:.5f} | "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.2f}%"
            f"{best_tag}"
        )

    print(f"\nTraining complete. Best validation accuracy: {best_val_acc:.2f}%")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train LeNet-5 CSNN (STBP) on N-MNIST")
    parser.add_argument("--epochs",    type=int,   default=20,                     help="Number of training epochs (default: 20)")
    parser.add_argument("--lr",        type=float, default=1e-3,                   help="Initial learning rate (default: 1e-3)")
    parser.add_argument("--batch",     type=int,   default=32,                     help="Batch size (default: 32)")
    parser.add_argument("--data-dir",  type=str,   default="preprocessed_data_native", help="Directory with preprocessed .npz files")
    args = parser.parse_args()

    train_snn(
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch,
        data_dir=args.data_dir,
    )
