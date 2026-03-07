"""
results_logger.py
=================
Structured results logging and export for reproducible experiments.

Manages per-epoch training metrics, per-sample evaluation data, and
ablation table generation. All outputs go to results/raw/ and results/processed/.
"""

import csv
import os
import json
from typing import Dict, List, Optional
from datetime import datetime


class ResultsLogger:
    """
    Structured logger that writes training metrics, evaluation results,
    and ablation comparison tables to CSV and JSON files.

    Usage:
        logger = ResultsLogger(experiment_name="sparse_full", seed=42)
        logger.log_epoch(epoch=1, train_acc=90.5, ...)
        logger.log_sample(sample_id=0, true_label=3, ...)
        logger.save_all()
    """

    def __init__(self, experiment_name: str = "experiment", seed: int = 42,
                 results_dir: str = "results"):
        self.experiment_name = experiment_name
        self.seed = seed
        self.results_dir = results_dir
        self.raw_dir = os.path.join(results_dir, "raw")
        self.processed_dir = os.path.join(results_dir, "processed")

        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

        # Epoch-level training metrics
        self._epoch_log: List[Dict] = []

        # Sample-level evaluation metrics
        self._sample_log: List[Dict] = []

        # Run-level summary
        self._run_summary: Optional[Dict] = None

        self._timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def log_epoch(self, epoch: int, **metrics):
        """
        Log metrics for a training epoch.

        Expected keys: train_loss, train_acc, val_loss, val_acc,
        avg_timesteps, total_spikes, gate_rejection_pct, sram_reads,
        mac_ops, lr, etc.
        """
        entry = {"epoch": epoch, "experiment": self.experiment_name,
                 "seed": self.seed}
        entry.update(metrics)
        self._epoch_log.append(entry)

    def log_sample(self, **metrics):
        """
        Log per-sample evaluation metrics.

        Expected keys: sample_id, true_label, predicted, correct,
        exit_timestep, sram_reads, gk_rejection_rate, total_spikes,
        firing_rate, energy_pj, confidence_final, etc.
        """
        entry = {"experiment": self.experiment_name, "seed": self.seed}
        entry.update(metrics)
        self._sample_log.append(entry)

    def set_run_summary(self, **metrics):
        """Set the final run-level summary (accuracy, best epoch, etc.)."""
        self._run_summary = {
            "experiment": self.experiment_name,
            "seed": self.seed,
            "timestamp": self._timestamp,
        }
        self._run_summary.update(metrics)

    def save_epoch_csv(self):
        """Export epoch log to CSV."""
        if not self._epoch_log:
            return
        path = os.path.join(self.raw_dir,
                            f"{self.experiment_name}_seed{self.seed}_epochs.csv")
        keys = list(self._epoch_log[0].keys())
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self._epoch_log)
        print(f"[Logger] Epoch log saved to {path}")

    def save_sample_csv(self):
        """Export sample log to CSV."""
        if not self._sample_log:
            return
        path = os.path.join(self.raw_dir,
                            f"{self.experiment_name}_seed{self.seed}_samples.csv")
        keys = list(self._sample_log[0].keys())
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self._sample_log)
        print(f"[Logger] Sample log saved to {path}")

    def save_run_summary(self):
        """Export run summary to JSON."""
        if not self._run_summary:
            return
        path = os.path.join(self.raw_dir,
                            f"{self.experiment_name}_seed{self.seed}_summary.json")
        with open(path, 'w') as f:
            json.dump(self._run_summary, f, indent=2)
        print(f"[Logger] Run summary saved to {path}")

    def save_all(self):
        """Save all logs."""
        self.save_epoch_csv()
        self.save_sample_csv()
        self.save_run_summary()

    @staticmethod
    def build_ablation_table(results_dir: str = "results",
                             output_name: str = "ablation_table.csv"):
        """
        Aggregate all run summaries in results/raw/ into an ablation table.

        Reads *_summary.json files, groups by experiment name, and computes
        mean ± std across seeds.
        """
        raw_dir = os.path.join(results_dir, "raw")
        processed_dir = os.path.join(results_dir, "processed")
        os.makedirs(processed_dir, exist_ok=True)

        summaries = []
        for fname in sorted(os.listdir(raw_dir)):
            if fname.endswith("_summary.json"):
                with open(os.path.join(raw_dir, fname)) as f:
                    summaries.append(json.load(f))

        if not summaries:
            print("[Logger] No summary files found for ablation table.")
            return

        # Group by experiment name
        from collections import defaultdict
        import numpy as np

        groups = defaultdict(list)
        for s in summaries:
            groups[s.get("experiment", "unknown")].append(s)

        # Build table rows
        rows = []
        metric_keys = ["accuracy", "avg_exit_timestep", "avg_sram_reads",
                        "gk_rejection_rate", "avg_firing_rate", "avg_energy_pj"]

        for exp_name, runs in sorted(groups.items()):
            row = {"experiment": exp_name, "num_seeds": len(runs)}
            for mk in metric_keys:
                values = [r.get(mk, 0) for r in runs if mk in r]
                if values:
                    arr = np.array(values, dtype=float)
                    row[f"{mk}_mean"] = float(arr.mean())
                    row[f"{mk}_std"] = float(arr.std())
                else:
                    row[f"{mk}_mean"] = 0.0
                    row[f"{mk}_std"] = 0.0
            rows.append(row)

        # Write CSV
        path = os.path.join(processed_dir, output_name)
        keys = list(rows[0].keys())
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)

        print(f"[Logger] Ablation table saved to {path}")
        return rows
