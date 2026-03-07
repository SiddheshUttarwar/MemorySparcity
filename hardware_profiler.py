"""
hardware_profiler.py
====================
Unified hardware accounting and energy estimation module.

Replaces scattered hw_* counter variables in the forward pass with a
single HardwareProfiler that tracks per-sample SRAM reads, MAC ops,
active neurons/synapses, and provides parameterized energy/latency estimates.

Reference RTL: Hardware_Architecture/*.v
"""

import csv
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional


@dataclass
class SampleProfile:
    """Hardware profile for a single inference sample."""
    sample_id: int = 0
    true_label: int = -1
    predicted_label: int = -1
    correct: bool = False

    # Temporal
    exit_timestep: int = 20
    T_max: int = 20

    # Gatekeeper
    gk_raw_spikes: int = 0
    gk_kept_spikes: int = 0
    gk_rejected: int = 0
    gk_rejection_rate: float = 0.0

    # SRAM access
    sram_reads_input: int = 0      # Reads triggered by input spikes (conv1)
    sram_reads_hidden: int = 0     # Reads triggered by internal spikes
    sram_reads_total: int = 0      # Total reads
    sram_writes: int = 0           # Writes (0 during inference)

    # Compute
    mac_ops: int = 0               # Total MAC operations
    total_spikes: int = 0          # Total spikes across all layers
    firing_rate: float = 0.0       # Mean firing rate across neurons

    # Per-layer spike counts
    spikes_layer1: int = 0
    spikes_layer2: int = 0
    spikes_layer3: int = 0
    spikes_layer4: int = 0

    # Active neurons per timestep (list)
    active_neurons_per_t: List[int] = field(default_factory=list)

    # Confidence trajectory (list of max-class probability per timestep)
    confidence_trajectory: List[float] = field(default_factory=list)

    # Cumulative SRAM reads per timestep
    cumulative_reads_per_t: List[int] = field(default_factory=list)

    # Energy estimates (pJ)
    energy_sram_pj: float = 0.0
    energy_mac_pj: float = 0.0
    energy_total_pj: float = 0.0

    # Latency estimate (ns)
    latency_ns: float = 0.0


class HardwareProfiler:
    """
    Accumulates hardware-relevant metrics during inference.

    Usage:
        profiler = HardwareProfiler(config)
        profiler.new_sample(sample_id, true_label)
        # ... during forward pass ...
        profiler.log_gatekeeper(raw, kept)
        profiler.log_layer_spikes(layer_idx, spike_count)
        profiler.log_sram_reads(input_reads, hidden_reads)
        profiler.log_timestep(t, active_neurons, cumulative_reads, confidence)
        # ... after forward pass ...
        profiler.finalize_sample(predicted, exit_t)
        profile = profiler.get_current_profile()
    """

    def __init__(self, sram_read_energy_pj=5.0, sram_write_energy_pj=5.0,
                 mac_energy_pj=1.0, clock_period_ns=10.0, T_max=20):
        self.sram_read_energy = sram_read_energy_pj
        self.sram_write_energy = sram_write_energy_pj
        self.mac_energy = mac_energy_pj
        self.clock_period = clock_period_ns
        self.T_max = T_max

        self._current: Optional[SampleProfile] = None
        self._profiles: List[SampleProfile] = []

    def new_sample(self, sample_id: int = 0, true_label: int = -1):
        """Start profiling a new sample."""
        self._current = SampleProfile(
            sample_id=sample_id,
            true_label=true_label,
            T_max=self.T_max,
        )

    def log_gatekeeper(self, raw_spikes: int, kept_spikes: int):
        """Accumulate gatekeeper statistics."""
        if self._current is None:
            return
        self._current.gk_raw_spikes += raw_spikes
        self._current.gk_kept_spikes += kept_spikes
        self._current.gk_rejected += (raw_spikes - kept_spikes)

    def log_sram_reads(self, input_reads: int, hidden_reads: int):
        """Log SRAM read events for this timestep."""
        if self._current is None:
            return
        self._current.sram_reads_input += input_reads
        self._current.sram_reads_hidden += hidden_reads

    def log_mac_ops(self, count: int):
        """Log MAC operation count."""
        if self._current is None:
            return
        self._current.mac_ops += count

    def log_layer_spikes(self, layer_idx: int, count: int):
        """Log spike count for a specific layer (0-indexed)."""
        if self._current is None:
            return
        if layer_idx == 0:
            self._current.spikes_layer1 += count
        elif layer_idx == 1:
            self._current.spikes_layer2 += count
        elif layer_idx == 2:
            self._current.spikes_layer3 += count
        elif layer_idx == 3:
            self._current.spikes_layer4 += count

    def log_timestep(self, t: int, active_neurons: int,
                     cumulative_reads: int, confidence: float = 0.0):
        """Log per-timestep dynamic metrics."""
        if self._current is None:
            return
        self._current.active_neurons_per_t.append(active_neurons)
        self._current.cumulative_reads_per_t.append(cumulative_reads)
        self._current.confidence_trajectory.append(confidence)

    def finalize_sample(self, predicted_label: int, exit_timestep: int,
                        total_spikes: float, total_neurons: int = 1):
        """Complete profiling for the current sample and compute derived metrics."""
        if self._current is None:
            return

        p = self._current
        p.predicted_label = predicted_label
        p.correct = (p.true_label == predicted_label)
        p.exit_timestep = exit_timestep
        p.total_spikes = int(total_spikes)
        p.sram_reads_total = p.sram_reads_input + p.sram_reads_hidden
        p.gk_rejection_rate = p.gk_rejected / max(1, p.gk_raw_spikes)
        p.firing_rate = total_spikes / max(1, total_neurons * exit_timestep)

        # Energy estimates
        p.energy_sram_pj = p.sram_reads_total * self.sram_read_energy
        p.energy_mac_pj = p.mac_ops * self.mac_energy
        p.energy_total_pj = p.energy_sram_pj + p.energy_mac_pj

        # Latency: timesteps × clock period
        p.latency_ns = exit_timestep * self.clock_period

        self._profiles.append(p)

    def get_current_profile(self) -> Optional[SampleProfile]:
        """Return the most recently finalized profile."""
        return self._profiles[-1] if self._profiles else None

    def get_all_profiles(self) -> List[SampleProfile]:
        """Return all collected profiles."""
        return self._profiles

    def aggregate(self) -> Dict:
        """Compute aggregate statistics across all profiled samples."""
        if not self._profiles:
            return {}

        import numpy as np
        n = len(self._profiles)

        def _stat(values):
            arr = np.array(values, dtype=float)
            return {"mean": float(arr.mean()), "std": float(arr.std()),
                    "min": float(arr.min()), "max": float(arr.max())}

        accuracy = sum(1 for p in self._profiles if p.correct) / n * 100

        return {
            "num_samples": n,
            "accuracy_pct": accuracy,
            "exit_timestep": _stat([p.exit_timestep for p in self._profiles]),
            "sram_reads_total": _stat([p.sram_reads_total for p in self._profiles]),
            "sram_reads_input": _stat([p.sram_reads_input for p in self._profiles]),
            "sram_reads_hidden": _stat([p.sram_reads_hidden for p in self._profiles]),
            "gk_rejection_rate": _stat([p.gk_rejection_rate for p in self._profiles]),
            "total_spikes": _stat([p.total_spikes for p in self._profiles]),
            "firing_rate": _stat([p.firing_rate for p in self._profiles]),
            "mac_ops": _stat([p.mac_ops for p in self._profiles]),
            "energy_total_pj": _stat([p.energy_total_pj for p in self._profiles]),
            "latency_ns": _stat([p.latency_ns for p in self._profiles]),
        }

    def export_csv(self, path: str):
        """Export all sample profiles to CSV."""
        if not self._profiles:
            return
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

        # Use scalar fields only for CSV
        scalar_fields = [
            'sample_id', 'true_label', 'predicted_label', 'correct',
            'exit_timestep', 'T_max',
            'gk_raw_spikes', 'gk_kept_spikes', 'gk_rejected', 'gk_rejection_rate',
            'sram_reads_input', 'sram_reads_hidden', 'sram_reads_total', 'sram_writes',
            'mac_ops', 'total_spikes', 'firing_rate',
            'spikes_layer1', 'spikes_layer2', 'spikes_layer3', 'spikes_layer4',
            'energy_sram_pj', 'energy_mac_pj', 'energy_total_pj', 'latency_ns',
        ]

        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=scalar_fields)
            writer.writeheader()
            for p in self._profiles:
                d = asdict(p)
                row = {k: d[k] for k in scalar_fields}
                writer.writerow(row)
        print(f"[Profiler] Exported {len(self._profiles)} profiles to {path}")

    def print_summary(self):
        """Print a formatted summary table."""
        agg = self.aggregate()
        if not agg:
            print("[Profiler] No samples profiled yet.")
            return

        print(f"\n{'='*60}")
        print(f"  Hardware Profile Summary ({agg['num_samples']} samples)")
        print(f"{'='*60}")
        print(f"  Accuracy:            {agg['accuracy_pct']:.2f}%")
        print(f"  Avg Exit Timestep:   {agg['exit_timestep']['mean']:.1f} ± {agg['exit_timestep']['std']:.1f}")
        print(f"  Avg SRAM Reads:      {agg['sram_reads_total']['mean']:.0f} ± {agg['sram_reads_total']['std']:.0f}")
        print(f"  Avg GK Rejection:    {agg['gk_rejection_rate']['mean']*100:.1f}%")
        print(f"  Avg Firing Rate:     {agg['firing_rate']['mean']*100:.2f}%")
        print(f"  Avg MAC Ops:         {agg['mac_ops']['mean']:.0f}")
        print(f"  Avg Energy:          {agg['energy_total_pj']['mean']:.0f} pJ")
        print(f"  Avg Latency:         {agg['latency_ns']['mean']:.0f} ns")
        print(f"{'='*60}\n")
