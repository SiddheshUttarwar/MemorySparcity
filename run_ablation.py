"""
run_ablation.py
===============
Automated ablation study runner.

Runs training + evaluation for each ablation configuration × multiple seeds,
collecting results into results/processed/ablation_table.csv.

Usage:
    python run_ablation.py                    # Full ablation (all configs × 3 seeds)
    python run_ablation.py --seeds 42         # Single seed
    python run_ablation.py --configs baseline full  # Specific configs only
    python run_ablation.py --eval-only        # Skip training, just evaluate
"""

import os
import sys
import argparse
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from experiment_config import ExperimentConfig, make_ablation_config, ABLATION_CONFIGS
from results_logger import ResultsLogger
from hardware_profiler import HardwareProfiler


def run_single_experiment(config: ExperimentConfig, train_loader, test_loader,
                          device, eval_only=False):
    """
    Run a single training + evaluation experiment with the given config.
    Returns a summary dict with all key metrics.
    """
    from sparse_snn_model import LeNet5_Sparse_CSNN

    config.set_seed()

    model = LeNet5_Sparse_CSNN(
        in_channels=config.in_channels,
        num_classes=config.num_classes,
        config=config,
    ).to(device)

    logger = ResultsLogger(
        experiment_name=config.name,
        seed=config.seed,
        results_dir=config.results_dir,
    )

    best_val_acc = 0.0
    best_epoch = 0

    if not eval_only:
        # --- Training ---
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs
        ) if config.scheduler == "cosine" else None

        criterion = nn.CrossEntropyLoss()

        print(f"\n{'='*60}")
        print(f"  Training: {config.name} (seed={config.seed})")
        print(f"  Ablation: {config.ablation_label()}")
        print(f"{'='*60}")

        for epoch in range(config.epochs):
            model.train()
            total_loss, correct, total = 0, 0, 0
            total_spikes = 0.0
            epoch_hw = {'cs': 0, 'mac': 0, 'in': 0, 'kept': 0}

            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                model.sync_from_sram()

                spike_rate, _, l1_spike_sum, actual_steps, hw_metrics = model(inputs)

                loss = criterion(spike_rate, targets)
                if config.use_sparsity_reg:
                    loss = loss + config.lambda_reg * l1_spike_sum

                loss.backward()
                optimizer.step()
                model.sync_to_sram()

                total_loss += loss.item()
                _, predicted = spike_rate.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                total_spikes += l1_spike_sum.item()
                epoch_hw['cs'] += hw_metrics['cs_asserts']
                epoch_hw['mac'] += hw_metrics['mac_ops']
                epoch_hw['in'] += hw_metrics['total_in']
                epoch_hw['kept'] += hw_metrics['kept_in']

            if scheduler:
                scheduler.step()

            train_acc = 100. * correct / total

            # Validation
            model.eval()
            model.sync_from_sram()
            val_correct, val_total = 0, 0
            val_spikes = 0.0
            val_steps_sum = 0

            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    spike_rate, _, l1, steps, _ = model(inputs)
                    _, predicted = spike_rate.max(1)
                    val_total += targets.size(0)
                    val_correct += predicted.eq(targets).sum().item()
                    val_spikes += l1.item()
                    val_steps_sum += steps

            val_acc = 100. * val_correct / val_total

            gate_kept_pct = 100. * epoch_hw['kept'] / max(1, epoch_hw['in'])

            logger.log_epoch(
                epoch=epoch + 1,
                train_loss=total_loss / len(train_loader),
                train_acc=train_acc,
                val_acc=val_acc,
                total_spikes=total_spikes,
                gate_kept_pct=gate_kept_pct,
                sram_reads=epoch_hw['cs'],
                mac_ops=epoch_hw['mac'],
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                ckpt_path = os.path.join(
                    config.results_dir, "raw",
                    f"{config.name}_seed{config.seed}_best.pth"
                )
                os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
                torch.save(model.state_dict(), ckpt_path)

            print(f"  Epoch {epoch+1}/{config.epochs} | "
                  f"Train: {train_acc:.1f}% | Val: {val_acc:.1f}% | "
                  f"GK Pass: {gate_kept_pct:.1f}%")

    # --- Evaluation with full profiling ---
    print(f"  Running full evaluation profiling...")
    model.eval()
    model.sync_from_sram()

    profiler = HardwareProfiler(
        sram_read_energy_pj=config.sram_read_energy_pj,
        mac_energy_pj=config.mac_energy_pj,
        clock_period_ns=config.clock_period_ns,
        T_max=config.T_max,
    )

    sample_id = 0
    eval_correct, eval_total = 0, 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            spike_rate, _, l1, steps, hw = model(inputs)
            _, predicted = spike_rate.max(1)

            for b in range(inputs.size(0)):
                profiler.new_sample(sample_id, targets[b].item())
                profiler.log_gatekeeper(hw['total_in'], hw['kept_in'])
                profiler.log_sram_reads(hw['kept_in'], hw['sram_reads_hidden'])
                profiler.log_mac_ops(hw['mac_ops'])
                for li in range(4):
                    profiler.log_layer_spikes(li, hw['per_layer_spikes'][li])
                profiler.finalize_sample(
                    predicted[b].item(), steps, l1.item() / inputs.size(0),
                    total_neurons=32*28*28 + 64*14*14 + 128 + 10
                )

                eval_correct += int(predicted[b].item() == targets[b].item())
                eval_total += 1
                sample_id += 1

    eval_acc = 100. * eval_correct / eval_total
    agg = profiler.aggregate()

    profiler.export_csv(os.path.join(
        config.results_dir, "raw",
        f"{config.name}_seed{config.seed}_hw_profiles.csv"
    ))
    profiler.print_summary()

    # Summary
    summary = {
        "accuracy": eval_acc,
        "best_epoch": best_epoch,
        "avg_exit_timestep": agg['exit_timestep']['mean'] if agg else 20,
        "avg_sram_reads": agg['sram_reads_total']['mean'] if agg else 0,
        "gk_rejection_rate": agg['gk_rejection_rate']['mean'] if agg else 0,
        "avg_firing_rate": agg['firing_rate']['mean'] if agg else 0,
        "avg_energy_pj": agg['energy_total_pj']['mean'] if agg else 0,
        "ablation_label": config.ablation_label(),
    }
    logger.set_run_summary(**summary)
    logger.save_all()

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run ablation studies")
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 7],
                        help='Random seeds to use')
    parser.add_argument('--configs', nargs='+', type=str,
                        default=list(ABLATION_CONFIGS.keys()),
                        help='Ablation configs to run')
    parser.add_argument('--eval-only', action='store_true',
                        help='Skip training, only run evaluation')
    parser.add_argument('--results-dir', type=str, default='results',
                        help='Results output directory')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    from train import NMNISTDataset
    train_dataset = NMNISTDataset(data_dir='preprocessed_data_native', split='train')
    test_dataset = NMNISTDataset(data_dir='preprocessed_data_native', split='test')
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print(f"\nAblation Study: {len(args.configs)} configs × {len(args.seeds)} seeds "
          f"= {len(args.configs) * len(args.seeds)} runs\n")

    all_results = []
    start = time.time()

    for config_name in args.configs:
        for seed in args.seeds:
            cfg = make_ablation_config(config_name, seed=seed)
            cfg.results_dir = args.results_dir

            result = run_single_experiment(
                cfg, train_loader, test_loader, device,
                eval_only=args.eval_only
            )
            result['config'] = config_name
            result['seed'] = seed
            all_results.append(result)

    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"  Ablation study complete! ({elapsed/60:.1f} minutes)")
    print(f"{'='*60}")

    # Build aggregated ablation table
    ResultsLogger.build_ablation_table(args.results_dir)

    # Print summary
    print(f"\n{'='*60}")
    print(f"  Summary Table")
    print(f"{'='*60}")
    print(f"  {'Config':<20} {'Seed':>5} {'Acc%':>7} {'Exit T':>7} {'SRAM Reads':>12} {'GK Rej%':>8}")
    print(f"  {'-'*60}")
    for r in all_results:
        print(f"  {r['config']:<20} {r['seed']:>5} {r['accuracy']:>6.1f}% "
              f"{r['avg_exit_timestep']:>7.1f} {r['avg_sram_reads']:>12.0f} "
              f"{r['gk_rejection_rate']*100:>7.1f}%")


if __name__ == '__main__':
    main()
