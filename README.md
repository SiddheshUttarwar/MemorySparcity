# Sparse Neuromorphic SNN: Software-Hardware Co-Design on N-MNIST

This repository implements a full **software-hardware co-design** pipeline: a sparsity-optimized **Convolutional Spiking Neural Network (CSNN)** trained in PyTorch with Surrogate Gradient Backpropagation (STBP), mapped to a custom **Verilog RTL** inference accelerator with spike-driven memory access.

### 📚 Documentation

| Document | Description |
|----------|-------------|
| **[mathematical_model.md](mathematical_model.md)** | All equations: LIF dynamics, surrogate gradient, gatekeeper, early exit, quantization, energy model |
| **[hardware_architecture.md](hardware_architecture.md)** | RTL block descriptions, signal tables, and parameters for all Verilog modules |
| **[lenet5.md](lenet5.md)** | Layer-by-layer shape transformations and parameter counts |
| **[lenet5_sparse_csnn_structure_table.md](lenet5_sparse_csnn_structure_table.md)** | Compact structural table (per timestep, shared weights) |
| **[hardware_pipeline_animation.html](hardware_pipeline_animation.html)** | Interactive animation: baseline vs sparse side-by-side with SRAM graph |

---

## 🌟 What Makes This Project Unique? (Our Novelty)

Unlike traditional AI tutorials, this project solves a real-world physical problem: **Memory Bottlenecks** (the "Memory Wall"). When AI runs on a physical chip, fetching data from memory (SRAM) consumes far more power than doing the actual math. 

We built a unique **Hardware-Software Pipeline** that attacks this problem from four different angles simultaneously:

1. **The "Bouncer" (Dynamic Gatekeeper):** Instead of processing every signal the camera sees, we built a hardware filter at the very front door. It ignores completely random noise and duplicate signals, stopping useless data before it ever touches the memory.
2. **"Lazy" Neurons (Adaptive Thresholds):** If a neuron fires too often (like a "spike storm"), it dynamically raises its own threshold, forcing itself to calm down and only fire when something truly important happens.
3. **The "Stop Early" Button (Temporal Early-Exit):** Instead of always examining the video for a fixed 20 frames, the network constantly checks its own confidence. If it recognizes a "7" clearly by frame 5, it shuts down the rest of the computation instantly, saving 75% of the energy.
4. **Hardware-Taught Software (Co-Design):** Usually, software AI is trained first and just "forced" to fit on hardware later. In our Python code, we explicitly track exactly how many times the simulated hardware fetches from memory. The AI actively learns to penalize itself for wasting memory during training.

The result is a highly accurate AI that uses a fraction of the memory bandwidth of standard models.

## 🧠 Model Architecture

The network follows a **LeNet-5** topology adapted for temporal spike processing. Each input sample is a sequence of **T=20 time bins** from the N-MNIST event camera (2 polarity channels, 28×28 spatial).

### Network Layers (Per Timestep)

| Layer | Type | Input | Output | Kernel | Parameters | Purpose |
|-------|------|-------|--------|--------|------------|---------|
| `conv1` | Conv2d | 2×28×28 | 32×28×28 | 5×5 | 1,600 | Spatial feature extraction |
| `bn1` | BatchNorm2d | 32×28×28 | 32×28×28 | — | 64 | Normalize activations before LIF threshold |
| `lif1` | Adaptive LIF | 32×28×28 | 32×28×28 (spikes) | — | 0 | Leaky integrate-and-fire with adaptive threshold |
| `pool1` | AvgPool2d | 32×28×28 | 32×14×14 | 2×2 | 0 | Downsample; AvgPool preserves firing rate density |
| `conv2` | Conv2d | 32×14×14 | 64×14×14 | 5×5 | 51,200 | Higher-level feature extraction |
| `bn2` | BatchNorm2d | 64×14×14 | 64×14×14 | — | 128 | Normalize before second LIF layer |
| `lif2` | Adaptive LIF | 64×14×14 | 64×14×14 (spikes) | — | 0 | Second spiking layer |
| `pool2` | AvgPool2d | 64×14×14 | 64×7×7 | 2×2 | 0 | Second spatial downsample |
| `flatten` | Flatten | 64×7×7 | 3136 | — | 0 | Reshape for fully-connected layers |
| `dropout` | Dropout(0.5) | 3136 | 3136 | — | 0 | Regularization (training only) |
| `fc1` | Linear | 3136 | 128 | — | 401,408 | Dense classification layer |
| `lif3` | Adaptive LIF | 128 | 128 (spikes) | — | 0 | Third spiking layer |
| `fc2` | Linear | 128 | 10 | — | 1,280 | Output layer (10 digit classes) |
| `lif4` | Adaptive LIF | 10 | 10 (spikes) | — | 0 | Output spike generation |

**Total Trainable Parameters: 455,680** (all weights are bias-free and INT8 quantized for SRAM)

### Key Layer Explanations

**Batch Normalization (BN):** Normalizes conv outputs to mean≈0, std≈1 before the LIF neuron. Without BN, the fixed threshold (`v_th=1.0`) would cause neurons to either never fire (values too small) or always fire (values too large). During inference, BN can be folded into conv weights for zero hardware overhead.

**Average Pooling (AvgPool):** Chosen over MaxPool because LIF outputs are binary spikes (0 or 1). MaxPool on binary data only tells "did any neuron fire?" — AvgPool preserves the **firing density** (e.g., 0.25 = 25% of neurons fired in that region), which carries much more information.

**Adaptive LIF Neuron:** Implements leaky integrate-and-fire dynamics with adaptive thresholding:
- Membrane potential: `v(t) = β·v(t-1) + x(t)` (leak factor β=0.9)
- Spike generation: fires when `v(t) > v_th(t)`
- Adaptive threshold: `v_th(t+1) = v_th(t) + ρ` after each spike (ρ=0.05), suppressing spike storms
- Uses **Fast Sigmoid surrogate gradient** for backpropagation through the non-differentiable spike function

---

## ⚡ Sparsity Optimizations

The Sparse CSNN (`sparse_snn_model.py`) implements four hardware-aware optimizations:

### 1. Dynamic Gatekeeper (Input Filter)
Filters incoming spike events before they trigger SRAM reads. Combines two sub-filters:
- **Importance Monitor:** Tracks per-pixel activity with saturating counters and a decay window. Low-activity events (background noise) are rejected.
- **Burst Redundancy Filter:** Suppresses consecutive identical spikes from the same source that add no new temporal information.
- **Result:** ~30% of input spikes are rejected → 30% fewer SRAM reads at the input layer.

### 2. Temporal Early Exit
Instead of always running T=20 timesteps, the forward pass monitors output-layer confidence via softmax probabilities. When confidence exceeds 90% for all samples in the batch, inference halts early (often at T=4–8 for easy digits), saving proportional compute and memory access.

### 3. Adaptive Thresholding
LIF neurons dynamically raise their firing threshold after spiking (by ρ=0.05 per spike). This creates **temporal sparsity** — neurons that fire frequently become harder to trigger, naturally reducing total spike count and downstream SRAM reads.

### 4. INT8 Weight Quantization
All weights are quantized to signed 8-bit integers (`-127` to `+127`) using symmetric quantization (`scale = max(|w|)/127`). This matches the 8-bit data width of the Verilog `quantized_sram.v` module and reduces SRAM storage by 4× compared to FP32.

---

## 🔧 Software-Hardware Bridge

The `export_weights_mem.py` script bridges training (software) and inference (hardware):

```
PyTorch Model (.pth) → INT8 Quantization → Hex .mem Files → Verilog $readmemh → quantized_sram.v
```

### Verilog Weight Loading
```verilog
quantized_sram #(
    .ADDR_WIDTH(11),
    .DATA_WIDTH(8),
    .MEM_FILE("mem_weights/conv1_weights.mem")  // Trained weights loaded here
) conv1_sram ( .clk(clk), .addr(addr), .re(spike_in), .data_out(weight) );
```

### Exported Weight Files

| File | Layer | Entries | SRAM Size |
|------|-------|---------|-----------|
| `conv1_weights.mem` | Conv1 (2×5×5 → 32) | 1,600 | 1.6 KB |
| `conv2_weights.mem` | Conv2 (32×5×5 → 64) | 51,200 | 50 KB |
| `fc1_weights.mem` | FC1 (3136 → 128) | 401,408 | 392 KB |
| `fc2_weights.mem` | FC2 (128 → 10) | 1,280 | 1.3 KB |

---

## 🚀 Quick Start (Google Colab — Recommended)

Since training SNNs on CPU can take days, use a free Colab GPU:

1. Open [Google Colab](https://colab.research.google.com/) → `Runtime → Change runtime type → T4 GPU`
2. Paste and run:

```bash
# Clone and setup
!git clone https://github.com/SiddheshUttarwar/MemorySparcity.git
%cd MemorySparcity
!unzip -q Train.zip && unzip -q Test.zip

# Preprocess 70,000 N-MNIST events
!python preprocess_dataset.py

# Install visualization tools
!pip install -q torchinfo torchviz graphviz

# Train baseline CSNN (saves best_baseline_model.pth)
!python train.py

# Train sparse CSNN (saves best_sparse_model.pth)
!python train_sparse.py

# Render architecture diagrams
!python visualize_model.py

# Run inference with hardware metrics
!python predict_sparse.py

# Compare baseline vs sparse (generates hardware_comparative_analysis.png)
!python predict_compare.py

# Export weights to Verilog .mem format
!python export_weights_mem.py --model best_sparse_model.pth
```

*(If `Train.zip`/`Test.zip` are not in the repo, drag-drop them from your computer into the Colab file browser before running `unzip`.)*

---

## 💻 Local Setup (Windows)

### Step 1: Dataset
Place `Train.zip` and `Test.zip` in the project root. Do not extract manually.

### Step 2: Python Environment
```powershell
python -m venv .venv
Set-ExecutionPolicy -ExecutionPolicy UNRESTRICTED -Scope CurrentUser
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy scipy matplotlib
```

### Step 3: Preprocess → Train → Evaluate
```powershell
python preprocess_dataset.py    # Preprocess 70,000 events (~2 min)
python train.py                 # Train baseline CSNN
python train_sparse.py          # Train sparse CSNN with HW metrics
python predict_sparse.py        # Single-sample inference
python predict_compare.py       # 100-sample baseline vs sparse comparison
python export_weights_mem.py    # Export weights for Verilog
```

---

## 📂 Project Structure

```
├── snn_model.py              # Baseline LeNet-5 CSNN (dense, no sparsity)
├── sparse_snn_model.py       # Sparse CSNN (gatekeeper + early exit + adaptive LIF)
├── SRAM.py                   # Simulated SRAM weight memory model
├── train.py                  # Baseline CSNN training script
├── train_sparse.py           # Sparse CSNN training with live HW metrics
├── predict_sparse.py         # Single-sample inference with SRAM read tracking
├── predict_compare.py        # 100-sample baseline vs sparse statistical comparison
├── export_weights_mem.py     # PyTorch → Verilog .mem INT8 weight export
├── preprocess_dataset.py     # N-MNIST raw events → spatial-temporal tensor pipeline
├── visualize_model.py        # Architecture diagram generation
├── Hardware_Architecture/    # Verilog RTL modules for inference accelerator
│   ├── sparse_snn_top.v      #   Top-level integration (connects all blocks)
│   ├── quantized_sram.v      #   INT8 SRAM with $readmemh MEM_FILE parameter
│   ├── sparse_mac.v          #   Spike-driven multiply-accumulate unit
│   ├── adaptive_lif.v        #   LIF neuron with adaptive threshold
│   ├── dynamic_gatekeeper.v  #   Input event filter (importance + burst)
│   ├── early_exit_fsm.v      #   Confidence-based early exit FSM
│   ├── importance_monitor.v  #   Density-based activity filter
│   ├── burst_redundancy.v    #   Consecutive spike suppressor
│   └── tb_sram_weights.v     #   Testbench for weight loading verification
├── mem_weights/              # Exported INT8 .mem files for Verilog $readmemh
├── hardware_architecture.md  # Detailed RTL block documentation
└── Neuromorphic_Report_ECE274.tex  # IEEE conference paper (LaTeX)
```

---

## 📄 Authors

- **Siddhesh Uttarwar** — University of California, Santa Barbara (suttarwar@ucsb.edu)
- **Parth Kulkarni** — University of California, Santa Barbara (parthkulkarni@ucsb.edu)
