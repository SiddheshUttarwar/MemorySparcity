"""
experiment_config.py
====================
Reproducible experiment configuration system with ablation flags.

Provides a dataclass-based config with YAML serialization, deterministic
seed fixing across torch/numpy/random/CUDA, and per-component ablation
toggles for systematic evaluation.

Usage:
    from experiment_config import ExperimentConfig
    cfg = ExperimentConfig.from_yaml("configs/sparse_full.yaml")
    cfg.set_seed()  # Fix all RNGs
"""

import os
import json
import random
import dataclasses
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import torch


@dataclass
class ExperimentConfig:
    """All hyperparameters and ablation flags for one experiment run."""

    # --- Experiment metadata ---
    name: str = "sparse_full"
    seed: int = 42
    description: str = ""

    # --- Model architecture ---
    in_channels: int = 2
    num_classes: int = 10
    beta: float = 0.9          # LIF leak factor
    v_threshold: float = 1.0   # Base firing threshold
    rho: float = 0.05          # Adaptive threshold step size

    # --- Training ---
    epochs: int = 10
    batch_size: int = 32
    lr: float = 2e-3
    lambda_reg: float = 1e-7   # L1 spike regularization weight
    dropout: float = 0.5
    scheduler: str = "cosine"  # "cosine" or "none"

    # --- Inference control ---
    T_max: int = 20            # Maximum time steps
    confidence_margin: float = 0.9  # Early-exit confidence threshold
    temperature: float = 5.0   # Softmax temperature for confidence

    # --- Gatekeeper parameters ---
    imp_thresh: float = 1.0    # Importance monitor threshold
    imp_win_tick: int = 5      # Window decay interval (timesteps)
    max_repeats: int = 1       # Burst redundancy max consecutive repeats

    # --- Ablation flags ---
    use_gatekeeper: bool = True
    use_early_exit: bool = True
    use_adaptive_threshold: bool = True
    use_quantization: bool = True
    use_sparsity_reg: bool = True

    # --- Hardware proxy parameters (for energy estimation) ---
    sram_read_energy_pj: float = 5.0    # Energy per SRAM read (pJ)
    sram_write_energy_pj: float = 5.0   # Energy per SRAM write (pJ)
    mac_energy_pj: float = 1.0          # Energy per MAC operation (pJ)
    clock_period_ns: float = 10.0       # Clock period (ns), 100 MHz

    # --- Paths ---
    data_dir: str = "preprocessed_data_native"
    results_dir: str = "results"
    checkpoint_path: str = "best_sparse_model.pth"

    def set_seed(self):
        """Fix all random number generators for reproducibility."""
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        print(f"[Config] Seed fixed to {self.seed}")

    def save_yaml(self, path: str):
        """Save config to YAML file."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        # Use simple key: value format (no PyYAML dependency)
        with open(path, 'w') as f:
            f.write(f"# Experiment: {self.name}\n")
            f.write(f"# {self.description}\n\n")
            for k, v in asdict(self).items():
                if isinstance(v, bool):
                    f.write(f"{k}: {'true' if v else 'false'}\n")
                elif isinstance(v, str):
                    f.write(f'{k}: "{v}"\n')
                else:
                    f.write(f"{k}: {v}\n")
        print(f"[Config] Saved to {path}")

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        """Load config from YAML file (simple key: value parser)."""
        values = {}
        type_hints = {f.name: f.type for f in dataclasses.fields(cls)}

        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if ':' not in line:
                    continue
                key, val = line.split(':', 1)
                key = key.strip()
                val = val.strip().strip('"').strip("'")

                if key not in type_hints:
                    continue

                expected = type_hints[key]
                if expected == bool or expected is bool:
                    values[key] = val.lower() in ('true', '1', 'yes')
                elif expected == int or expected is int:
                    values[key] = int(val)
                elif expected == float or expected is float:
                    values[key] = float(val)
                else:
                    values[key] = val

        return cls(**values)

    def save_json(self, path: str):
        """Save config as JSON (for programmatic consumption)."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    def ablation_label(self) -> str:
        """Generate a short label describing which ablations are active."""
        parts = []
        if self.use_gatekeeper: parts.append("GK")
        if self.use_early_exit: parts.append("EE")
        if self.use_adaptive_threshold: parts.append("AT")
        if self.use_quantization: parts.append("Q8")
        if self.use_sparsity_reg: parts.append("SR")
        return "+".join(parts) if parts else "Baseline"

    def __str__(self):
        label = self.ablation_label()
        return (f"ExperimentConfig(name={self.name}, seed={self.seed}, "
                f"ablation=[{label}], epochs={self.epochs}, lr={self.lr})")


# --- Preset factory functions ---

def baseline_config(seed: int = 42) -> ExperimentConfig:
    """Dense CSNN baseline with all optimizations disabled."""
    return ExperimentConfig(
        name="baseline",
        seed=seed,
        description="Dense CSNN baseline - no sparsity optimizations",
        use_gatekeeper=False,
        use_early_exit=False,
        use_adaptive_threshold=False,
        use_quantization=False,
        use_sparsity_reg=False,
        checkpoint_path="best_baseline_model.pth",
    )


def sparse_full_config(seed: int = 42) -> ExperimentConfig:
    """Full sparsity-optimized CSNN with all components enabled."""
    return ExperimentConfig(
        name="sparse_full",
        seed=seed,
        description="Full sparse CSNN with gatekeeper + early exit + adaptive threshold",
    )


ABLATION_CONFIGS = {
    "baseline": {"use_gatekeeper": False, "use_early_exit": False,
                 "use_adaptive_threshold": False, "use_sparsity_reg": False},
    "gatekeeper_only": {"use_gatekeeper": True, "use_early_exit": False,
                        "use_adaptive_threshold": False, "use_sparsity_reg": False},
    "early_exit_only": {"use_gatekeeper": False, "use_early_exit": True,
                        "use_adaptive_threshold": False, "use_sparsity_reg": False},
    "adaptive_th_only": {"use_gatekeeper": False, "use_early_exit": False,
                         "use_adaptive_threshold": True, "use_sparsity_reg": False},
    "sparsity_reg_only": {"use_gatekeeper": False, "use_early_exit": False,
                          "use_adaptive_threshold": False, "use_sparsity_reg": True},
    "no_gatekeeper": {"use_gatekeeper": False, "use_early_exit": True,
                      "use_adaptive_threshold": True, "use_sparsity_reg": True},
    "no_early_exit": {"use_gatekeeper": True, "use_early_exit": False,
                      "use_adaptive_threshold": True, "use_sparsity_reg": True},
    "full": {"use_gatekeeper": True, "use_early_exit": True,
             "use_adaptive_threshold": True, "use_sparsity_reg": True},
}


def make_ablation_config(ablation_name: str, seed: int = 42) -> ExperimentConfig:
    """Create a config for a specific ablation variant."""
    if ablation_name not in ABLATION_CONFIGS:
        raise ValueError(f"Unknown ablation: {ablation_name}. "
                         f"Available: {list(ABLATION_CONFIGS.keys())}")
    flags = ABLATION_CONFIGS[ablation_name]
    return ExperimentConfig(
        name=f"ablation_{ablation_name}",
        seed=seed,
        description=f"Ablation study: {ablation_name}",
        **flags,
    )
