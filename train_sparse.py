"""
train_sparse.py
===============
Config-driven training script for the Sparse CSNN.

Usage:
    python train_sparse.py                                    # Default config
    python train_sparse.py --config configs/sparse_full.yaml  # From YAML
    python train_sparse.py --seed 123                         # Override seed
"""

import os
import argparse
import time
import torch
import torch.nn as nn
try:
    from torchinfo import summary
except ImportError:
    summary = None
from torch.utils.data import DataLoader
from train import NMNISTDataset
from sparse_snn_model import LeNet5_Sparse_CSNN
from experiment_config import ExperimentConfig
from results_logger import ResultsLogger


def train_sparse(config=None):
    if config is None:
        config = ExperimentConfig()

    config.set_seed()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing SPARSE STBP Training on Device: {DEVICE}")
    print(f"Config: {config}")

    train_dataset = NMNISTDataset(data_dir=config.data_dir, split='train')
    test_dataset = NMNISTDataset(data_dir=config.data_dir, split='test')

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    if len(train_dataset) == 0:
        print("No training data found. Make sure preprocess_dataset.py completed successfully.")
        return

    # Initialize Sparse Model with config
    model = LeNet5_Sparse_CSNN(
        in_channels=config.in_channels,
        num_classes=config.num_classes,
        config=config,
    ).to(DEVICE)

    # Optional Visualization
    if summary is not None:
        print("\n--- Model Architecture Visualization ---")
        summary(model, input_size=(1, config.T_max, config.in_channels, 28, 28), device=DEVICE)
        print("--------------------------------------\n")

    # Results logger
    logger = ResultsLogger(
        experiment_name=config.name,
        seed=config.seed,
        results_dir=config.results_dir,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = None
    if config.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    criterion = nn.CrossEntropyLoss()

    print("\n--- Starting SPARSE CSNN Training (STBP) ---")
    print(f"  Ablation: {config.ablation_label()}")
    start_time_global = time.time()

    best_val_acc = 0.0

    for epoch in range(config.epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        total_network_spikes = 0.0
        total_time_steps = 0
        epoch_hw_cs = 0
        epoch_hw_mac = 0
        epoch_hw_in = 0
        epoch_hw_kept = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            optimizer.zero_grad()
            model.sync_from_sram()

            # Forward pass
            spike_rate, output_spikes, l1_spike_sum, actual_steps, hw_metrics = model(inputs)

            loss_task = criterion(spike_rate, targets)
            # Apply L1 Spike Regularization (controlled by ablation flag)
            if config.use_sparsity_reg:
                loss_reg = config.lambda_reg * l1_spike_sum
                loss = loss_task + loss_reg
            else:
                loss = loss_task

            loss.backward()
            optimizer.step()
            model.sync_to_sram()

            total_loss += loss.item()
            _, predicted = spike_rate.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            total_network_spikes += l1_spike_sum.item()
            total_time_steps += actual_steps

            epoch_hw_cs += hw_metrics['cs_asserts']
            epoch_hw_mac += hw_metrics['mac_ops']
            epoch_hw_in += hw_metrics['total_in']
            epoch_hw_kept += hw_metrics['kept_in']

        train_acc = 100. * correct / total
        train_loss = total_loss / len(train_loader)

        spikes_per_inference_train = total_network_spikes / total
        sram_efficiency = train_acc / (spikes_per_inference_train / 1000.0 + 1e-9)
        avg_t = total_time_steps / len(train_loader)

        # Validation
        model.eval()
        model.sync_from_sram()
        test_loss, test_correct, test_total = 0, 0, 0
        val_network_spikes = 0.0
        val_time_steps = 0

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                spike_rate, output_spikes, l1_spike_sum, actual_steps, hw_metrics = model(inputs)

                loss = criterion(spike_rate, targets)
                test_loss += loss.item()

                _, predicted = spike_rate.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()
                val_network_spikes += l1_spike_sum.item()
                val_time_steps += actual_steps

        val_acc = 100. * test_correct / test_total
        val_loss = test_loss / len(test_loader)
        spikes_per_inference_val = val_network_spikes / test_total
        val_sram_efficiency = val_acc / (spikes_per_inference_val / 1000.0 + 1e-9)
        val_avg_t = val_time_steps / len(test_loader)

        gate_sparsity = 100. * (epoch_hw_kept / (epoch_hw_in + 1e-9))

        if scheduler:
            scheduler.step()

        current_lr = scheduler.get_last_lr()[0] if scheduler else config.lr

        # Log to CSV
        logger.log_epoch(
            epoch=epoch + 1,
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
            avg_timesteps_train=avg_t,
            avg_timesteps_val=val_avg_t,
            total_spikes_train=total_network_spikes,
            total_spikes_val=val_network_spikes,
            spikes_per_inference_train=spikes_per_inference_train,
            spikes_per_inference_val=spikes_per_inference_val,
            sram_efficiency_train=sram_efficiency,
            sram_efficiency_val=val_sram_efficiency,
            gate_kept_pct=gate_sparsity,
            sram_reads=epoch_hw_cs,
            mac_ops=epoch_hw_mac,
            lr=current_lr,
        )

        print(f"\nEpoch [{epoch+1}/{config.epochs}] | LR: {current_lr:.6f}")
        print(f"  Train -> Loss: {train_loss:.4f} | Acc: {train_acc:.2f}% | T-Avg: {avg_t:.1f} steps | SRAM Eff: {sram_efficiency:.4f}")
        print(f"  Test  -> Loss: {val_loss:.4f} | Acc: {val_acc:.2f}% | T-Avg: {val_avg_t:.1f} steps | SRAM Eff: {val_sram_efficiency:.4f}")
        print(f"  Spikes-> Train Total: {total_network_spikes:,.0f} | Val Total: {val_network_spikes:,.0f}")
        print(f"  HW Sim-> Input Gate Kept: {gate_sparsity:.1f}% ({epoch_hw_kept:,.0f}/{epoch_hw_in:,.0f}) | CS_asserts: {epoch_hw_cs:,.0f} | MAC_ops: {epoch_hw_mac:,.0f}")

        # Checkpoint the Best Model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"  => [Saving Checkpoint]: New Best Accuracy {best_val_acc:.2f}%")
            torch.save(model.state_dict(), config.checkpoint_path)

    elapsed = time.time() - start_time_global

    # Save run summary
    logger.set_run_summary(
        accuracy=best_val_acc,
        best_epoch=config.epochs,
        total_time_seconds=elapsed,
        ablation_label=config.ablation_label(),
    )
    logger.save_all()

    # Save config alongside results
    config.save_yaml(os.path.join(config.results_dir, "raw",
                                   f"{config.name}_seed{config.seed}_config.yaml"))

    print(f"\nSparse Training Completed in {elapsed:.2f} seconds!")
    print(f"Best Validation Accuracy achieved: {best_val_acc:.2f}%")
    print(f"Model saved to: {config.checkpoint_path}")
    print(f"Results saved to: {config.results_dir}/raw/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Sparse CSNN")
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML config file')
    parser.add_argument('--seed', type=int, default=None,
                        help='Override random seed')
    args = parser.parse_args()

    if args.config:
        cfg = ExperimentConfig.from_yaml(args.config)
    else:
        cfg = ExperimentConfig()

    if args.seed is not None:
        cfg.seed = args.seed

    train_sparse(cfg)
