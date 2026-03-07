"""
gatekeeper.py
=============
Formalized Dynamic Gatekeeper Controller for spike event filtering.

Extracted from the inline logic in sparse_snn_model.py and implemented as
a dedicated nn.Module with clean interface, per-timestep logging, and
summary statistics.

The gatekeeper combines two sub-filters:
  1. ImportanceMonitor  — density-based filter using saturating counters
  2. BurstRedundancyFilter — suppresses consecutive duplicate spike events

Reference RTL: Hardware_Architecture/dynamic_gatekeeper.v
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class GatekeeperStepMetrics:
    """Metrics captured at a single timestep."""
    timestep: int = 0
    raw_spikes: int = 0         # Total incoming spike events
    imp_rejected: int = 0       # Rejected by importance monitor
    burst_rejected: int = 0     # Rejected by burst filter
    kept: int = 0               # Events that passed through
    rejection_rate: float = 0.0 # Fraction rejected this step


class GatekeeperController(nn.Module):
    """
    Dynamic Gatekeeper: filters low-value and redundant spike events
    before they trigger downstream SRAM reads and MAC operations.

    Interface:
        gate_keep, metrics = gatekeeper(x_raw_t, t)

    When disabled (bypass mode): returns all spikes unfiltered.

    Args:
        channels: Number of input channels (C)
        height: Spatial height (H)
        width: Spatial width (W)
        imp_thresh: Minimum counter value for importance (default: 1.0)
        win_tick: Counter decay interval in timesteps (default: 5)
        max_repeats: Maximum consecutive repeats before burst rejection (default: 1)
        enabled: Whether gatekeeper is active (False = bypass, all spikes pass)
    """

    def __init__(self, channels=2, height=28, width=28,
                 imp_thresh=1.0, win_tick=5, max_repeats=1, enabled=True):
        super().__init__()
        self.channels = channels
        self.height = height
        self.width = width
        self.imp_thresh = imp_thresh
        self.win_tick = win_tick
        self.max_repeats = max_repeats
        self.enabled = enabled

        # Per-run log of step metrics
        self._step_log: List[GatekeeperStepMetrics] = []

        # Cumulative counters for the current inference
        self._total_raw = 0
        self._total_kept = 0
        self._total_imp_rejected = 0
        self._total_burst_rejected = 0

    def reset_state(self, batch_size: int, device: torch.device):
        """Initialize gatekeeper state for a new inference pass."""
        B, C, H, W = batch_size, self.channels, self.height, self.width

        # Importance monitor counters
        self.imp_cnt = torch.zeros(B, C, H, W, device=device)

        # Burst redundancy state
        self.last_spiked = torch.zeros(B, C, H, W, dtype=torch.bool, device=device)
        self.repeat_count = torch.zeros(B, C, H, W, device=device)

        # Clear logs
        self._step_log = []
        self._total_raw = 0
        self._total_kept = 0
        self._total_imp_rejected = 0
        self._total_burst_rejected = 0

    def forward(self, x_raw_t: torch.Tensor, t: int) -> tuple:
        """
        Process one timestep of raw input spikes.

        Args:
            x_raw_t: Raw input tensor [B, C, H, W]
            t: Current timestep index

        Returns:
            x_filtered: Filtered input (same shape as x_raw_t)
            step_metrics: GatekeeperStepMetrics for this timestep
        """
        is_spike = (x_raw_t > 0)
        raw_count = is_spike.sum().item()

        if not self.enabled:
            # Bypass mode: pass everything through
            metrics = GatekeeperStepMetrics(
                timestep=t, raw_spikes=int(raw_count),
                kept=int(raw_count), rejection_rate=0.0
            )
            self._total_raw += int(raw_count)
            self._total_kept += int(raw_count)
            self._step_log.append(metrics)
            return x_raw_t, metrics

        # --- 1. Importance Monitor ---
        self.imp_cnt = self.imp_cnt + x_raw_t
        if t > 0 and t % self.win_tick == 0:
            self.imp_cnt = torch.floor(self.imp_cnt / 2.0)  # Bit-shift decay

        imp_keep = (self.imp_cnt >= self.imp_thresh)
        imp_rejected_count = (is_spike & ~imp_keep).sum().item()

        # --- 2. Burst Redundancy Filter ---
        self.repeat_count = torch.where(
            is_spike & self.last_spiked,
            self.repeat_count + 1,
            torch.zeros_like(self.repeat_count)
        )
        self.last_spiked = is_spike
        corr_keep = (self.repeat_count <= self.max_repeats)
        burst_rejected_count = (is_spike & imp_keep & ~corr_keep).sum().item()

        # --- 3. Combined decision ---
        gate_keep = is_spike & imp_keep & corr_keep
        kept_count = gate_keep.sum().item()

        # Apply gate: zero out rejected spikes
        x_filtered = gate_keep.float() * x_raw_t

        # --- Log metrics ---
        rejection_rate = 1.0 - (kept_count / max(1, raw_count))
        metrics = GatekeeperStepMetrics(
            timestep=t,
            raw_spikes=int(raw_count),
            imp_rejected=int(imp_rejected_count),
            burst_rejected=int(burst_rejected_count),
            kept=int(kept_count),
            rejection_rate=rejection_rate,
        )
        self._step_log.append(metrics)
        self._total_raw += int(raw_count)
        self._total_kept += int(kept_count)
        self._total_imp_rejected += int(imp_rejected_count)
        self._total_burst_rejected += int(burst_rejected_count)

        return x_filtered, metrics

    def get_summary(self) -> Dict:
        """Return aggregate statistics for the current inference pass."""
        total_rejected = self._total_raw - self._total_kept
        return {
            "total_raw_spikes": self._total_raw,
            "total_kept": self._total_kept,
            "total_rejected": total_rejected,
            "rejection_rate": total_rejected / max(1, self._total_raw),
            "imp_rejected": self._total_imp_rejected,
            "burst_rejected": self._total_burst_rejected,
            "num_timesteps": len(self._step_log),
            "per_step_kept": [m.kept for m in self._step_log],
            "per_step_rejection_rate": [m.rejection_rate for m in self._step_log],
        }

    def get_step_log(self) -> List[GatekeeperStepMetrics]:
        """Return the full per-timestep log."""
        return self._step_log
