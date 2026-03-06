import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from train import NMNISTDataset
from snn_model import LeNet5_CSNN
from sparse_snn_model import LeNet5_Sparse_CSNN

class LIFHook:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.spikes_per_t = None

    def hook_fn(self, module, input, output):
        # Output shape from LIFNodeSTBP is [B, T, ...]
        # We want to sum over all dimensions except T
        dims_to_sum = list(range(output.dim()))
        dims_to_sum.remove(1) # Keep time dimension
        self.spikes_per_t = output.sum(dim=dims_to_sum).detach().cpu().numpy()

    def close(self):
        self.hook.remove()

def predict_compare():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "="*60)
    print(f"LOADING HARDWARE COMPARISON SUITE ({DEVICE})")
    print("="*60)

    # 1. Initialize Both Models
    baseline_model = LeNet5_CSNN(in_channels=2, num_classes=10).to(DEVICE)
    sparse_model = LeNet5_Sparse_CSNN(in_channels=2, num_classes=10).to(DEVICE)
    
    # 2. Load Weights
    try:
        baseline_model.load_state_dict(torch.load("best_baseline_model.pth", map_location=DEVICE))
        print("✅ Baseline Model Loaded")
    except FileNotFoundError:
        print("⚠️ Warning: 'best_baseline_model.pth' not found. Using untrained baseline for comparison.")

    try:
        sparse_model.load_state_dict(torch.load("best_sparse_model.pth", map_location=DEVICE))
        print("✅ Sparse Model Loaded")
    except FileNotFoundError:
        print("❌ Error: 'best_sparse_model.pth' not found.")
        return

    baseline_model.eval()
    sparse_model.eval()

    # Pre-sync the PyTorch weights into simulated Quantized SRAM banks
    baseline_model.sync_to_sram()
    baseline_model.sync_from_sram()
    
    sparse_model.sync_to_sram()
    sparse_model.sync_from_sram()

    # 3. Load Testing Data
    test_dataset = NMNISTDataset(data_dir='preprocessed_data_native', split='test')
    
    # 4. Grab 100 Random Samples for Statistical Significance
    num_samples = 100
    indices = random.sample(range(len(test_dataset)), num_samples)

    print("\n" + "="*60)
    print(f"EXECUTING INFERENCES ({num_samples} SAMPLES)")
    print("="*60)

    # Attach hooks to baseline LIF nodes
    hooks = [
        LIFHook(baseline_model.lif1),
        LIFHook(baseline_model.lif2),
        LIFHook(baseline_model.lif3),
        LIFHook(baseline_model.lif4)
    ]

    baseline_reads_list = []
    sparse_reads_list = []
    savings_list = []
    
    # Advanced Statistics Tracking
    sparse_steps_list = []
    baseline_firing_rates = []
    sparse_firing_rates = []
    gatekeeper_rejections = []
    
    # Total internal SNN neurons (Conv1 + Conv2 + FC1 + FC2)
    # 32x28x28 + 64x14x14 + 128 + 10 = 25088 + 12544 + 138 = 37770 neurons
    TOTAL_NEURONS = 37770
    
    # Track the step-wise data for one representative sample to graph
    rep_baseline_cumulative = []
    rep_sparse_cumulative = []
    rep_sample_idx = 0

    for i, idx in enumerate(indices):
        x_seq, label = test_dataset[idx]
        x_seq = x_seq.unsqueeze(0).to(DEVICE)  

        # ---------------------------------------------------------
        # A. BASELINE INFERENCE
        # ---------------------------------------------------------
        with torch.no_grad():
            baseline_model(x_seq)
            
        # Time steps = 20
        T = x_seq.size(1)
        
        # Calculate Input Spikes per TimeStep
        dims_to_sum = list(range(x_seq.dim()))
        dims_to_sum.remove(1)
        input_spikes_per_t = x_seq.sum(dim=dims_to_sum).detach().cpu().numpy()
        
        # Aggregate internal spikes per TimeStep
        internal_spikes_per_t = np.zeros(T)
        for h in hooks:
            internal_spikes_per_t += h.spikes_per_t
            
        total_spikes_per_t = input_spikes_per_t + internal_spikes_per_t
        baseline_cumulative = np.cumsum(total_spikes_per_t)
        baseline_total_reads = int(baseline_cumulative[-1])

        # ---------------------------------------------------------
        # B. SPARSE INFERENCE
        # ---------------------------------------------------------
        with torch.no_grad():
            spike_rate, output_spikes, l1_spike_sum, actual_steps, hw_metrics = sparse_model(x_seq, early_exit=True)

        sparse_total_reads = hw_metrics['kept_in'] + int(l1_spike_sum.item())
        sparse_cumulative = hw_metrics['cumulative_reads_over_time']
        
        # Extend the sparse cumulative list to T=20 if early exit occurred to match baseline length for graph
        sparse_cumulative_extended = list(sparse_cumulative)
        while len(sparse_cumulative_extended) < T:
            sparse_cumulative_extended.append(sparse_cumulative[-1])

        # ---------------------------------------------------------
        # C. RECORD STATS
        # ---------------------------------------------------------
        baseline_reads_list.append(baseline_total_reads)
        sparse_reads_list.append(sparse_total_reads)
        
        saving_pct = 100 * (1.0 - (sparse_total_reads / max(1, baseline_total_reads)))
        savings_list.append(saving_pct)
        
        # Advanced Stats
        sparse_steps_list.append(actual_steps)
        
        # Firing Rate = spikes / (neurons * time_steps)
        b_firing_rate = (internal_spikes_per_t.sum()) / (TOTAL_NEURONS * T)
        s_firing_rate = (l1_spike_sum.item()) / (TOTAL_NEURONS * actual_steps)
        baseline_firing_rates.append(b_firing_rate)
        sparse_firing_rates.append(s_firing_rate)
        
        reject_pct = 100 * (1.0 - (hw_metrics['kept_in'] / max(1, hw_metrics['total_in'])))
        gatekeeper_rejections.append(reject_pct)
        
        # Save representational array for plotting
        if i == 0:
            rep_baseline_cumulative = list(baseline_cumulative)
            rep_sparse_cumulative = sparse_cumulative_extended
            rep_sample_idx = idx

    for h in hooks:
        h.close()

    # ========================================================
    # STATISTICAL ANALYSIS
    # ========================================================
    mean_baseline = np.mean(baseline_reads_list)
    mean_sparse = np.mean(sparse_reads_list)
    mean_savings = np.mean(savings_list)
    
    mean_b_fire = np.mean(baseline_firing_rates) * 100 # Convert to percentage
    mean_s_fire = np.mean(sparse_firing_rates) * 100
    mean_gate_reject = np.mean(gatekeeper_rejections)
    mean_sparse_t = np.mean(sparse_steps_list)
    early_exit_count = sum(1 for steps in sparse_steps_list if steps < 20)

    print(f"\nAdvanced Statistical Analysis across {num_samples} Inference Targets:")
    print("--- 🧠 NEURAL METRICS ---")
    print(f"  Baseline Avg Firing Rate: {mean_b_fire:.3f}% of neurons active per step")
    print(f"  Sparse Avg Firing Rate  : {mean_s_fire:.3f}% of neurons active per step")
    print("--- ⚡ HARDWARE METRICS ---")
    print(f"  Avg Gatekeeper Rejection: {mean_gate_reject:.1f}% of sensory spikes dropped")
    print(f"  Avg Inference Time (T)  : {mean_sparse_t:.1f} steps (Baseline always 20.0)")
    print(f"  Early-Exit Frequency    : {early_exit_count}/{num_samples} inferences stopped early")
    print("--- 💾 MEMORY FETCHES ---")
    print(f"  Avg Baseline SRAM Reads : {mean_baseline:,.0f} per inference")
    print(f"  Avg Sparse SRAM Reads   : {mean_sparse:,.0f} per inference")
    print(f"  Mean Hardware Savings   : {mean_savings:.2f}% (StdDev: {np.std(savings_list):.2f}%)")

    # ========================================================
    # VISUALIZATION
    # ========================================================
    plt.figure(figsize=(18, 6))

    # Plot 1: Cumulative Reads Over Time (Representative Sample)
    plt.subplot(1, 3, 1)
    t_axis = list(range(1, 21))
    plt.plot(t_axis, rep_baseline_cumulative, color='#e74c3c', linewidth=2.5, marker='x', label="Baseline CSNN")
    plt.plot(t_axis, rep_sparse_cumulative, color='#2ecc71', linewidth=2.5, marker='o', label="Sparse CSNN (Gatekeeper+Early Exit)")
    
    plt.title(f"Memory Dynamics (Sample ID: {rep_sample_idx})", fontsize=12)
    plt.xlabel("Execution Time Steps (t)", fontsize=11)
    plt.ylabel("Cumulative SRAM Memory Fetches", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()

    # Plot 2: Average Firing Rate Comparison
    plt.subplot(1, 3, 2)
    categories = ['Baseline SNN', 'Sparse SNN']
    fire_values = [mean_b_fire, mean_s_fire]
    colors_fire = ['#9b59b6', '#3498db']
    
    bars2 = plt.bar(categories, fire_values, color=colors_fire, alpha=0.9, width=0.5)
    plt.title(f"Mean Neuronal Firing Rate", fontsize=12)
    plt.ylabel("% of Neurons Active per Time Step", fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    for bar in bars2:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + (max(fire_values)*0.02), f'{yval:.3f}%', ha='center', va='bottom', fontweight='bold')

    # Plot 3: Average Reads Bar Chart
    plt.subplot(1, 3, 3)
    values = [mean_baseline, mean_sparse]
    colors_reads = ['#e74c3c', '#2ecc71']
    
    # Plot 3: Average Reads Bar Chart
    bars3 = plt.bar(categories, values, color=colors_reads, alpha=0.9, width=0.5)
    plt.title(f"Average Hardware Cost ({num_samples} Samples)", fontsize=12)
    plt.ylabel("Average SRAM Reads per Inference", fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Add text labels on bars
    for bar in bars3:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + (max(values)*0.02), f'{int(yval):,}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    
    graph_filename = "hardware_comparative_analysis.png"
    plt.savefig(graph_filename, dpi=300, bbox_inches='tight')
    
    print("\n" + "="*60)
    print(f"Results successfully plotted and saved to: {graph_filename}")
    print("="*60)

if __name__ == "__main__":
    predict_compare()
