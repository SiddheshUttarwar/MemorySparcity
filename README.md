<<<<<<< HEAD
# STDPMemorySparcity
Design of a Event Based Sparse SRAM for STDP LUT implementation
=======
# ECE274 Neuromorphic Computing Project

This project builds an event-based SNN pipeline on N-MNIST with:
- LUT-based STDP learning
- correlation-based neuron gating
- SRAM-style synaptic memory modeling
- baseline and sparse-index architecture variants
- memory-read and weight-change analysis tools

## Requirements
- Python 3.x
- `numpy`
- `matplotlib`

Install:
```bash
pip install numpy matplotlib
```

## Dataset
Expected files in repo root:
- `Train.zip`
- `Test.zip`

N-MNIST event format (`.bin`):
- 5 bytes/event
- byte1=`x`, byte2=`y`
- byte3 MSB=`polarity` (`0` OFF, `1` ON)
- remaining 23 bits=`timestamp (us)`

Reference files:
- `ReadMe(MNIST).txt`
- `Read_Ndataset.m`

## Project Flow (Recommended)
1. Preprocess train/test data
2. Visualize data sanity (raw or preprocessed)
3. Train baseline architecture
4. Train sparse-index architecture
5. Compare memory reads and weight changes

## 1) Preprocess Data
Script: `preprocess_dataset.py`

```bash
python preprocess_dataset.py --zip-path Train.zip --output-dir preprocessed_train --time-bins 10 --normalize max
python preprocess_dataset.py --zip-path Test.zip --output-dir preprocessed_test --time-bins 10 --normalize max
```

Output:
- `preprocessed_train/*.npz`
- `preprocessed_test/*.npz`
- per-folder `manifest.json`

## 2) Visualization Scripts

### 2.1 Raw sample visualization
Script: `visualize_dataset.py`

```bash
python visualize_dataset.py --zip-path Train.zip --label 3 --sample-index 10
```

### 2.2 Preprocessed dataset summary
Script: `visualize_preprocessed_training.py`

```bash
python visualize_preprocessed_training.py --data-dir preprocessed_train --max-samples 2000 --sample-mode stratified
```

### 2.3 ON/OFF maps for digits 0-9 (from raw zip)
Script: `visualize_all_digits_onoff.py`

```bash
python visualize_all_digits_onoff.py --zip-path Train.zip --samples-per-digit 100 --save-fig reports/all_digits_onoff.png
```

### 2.4 ON/OFF maps for digits 0-9 (from preprocessed `.npz`)
Script: `visualize_preprocessed_onoff_0_9.py`

```bash
python visualize_preprocessed_onoff_0_9.py --data-dir preprocessed_train --samples-per-digit 50 --save-fig reports/preprocessed_onoff_0_9.png
```

### 2.5 Test + visualize the first trained NumPy model
Script: `test_visualize_first_model.py`

```bash
python test_visualize_first_model.py --model-path checkpoints/snn_numpy_smoke.npz --data-dir preprocessed_test --save-fig reports/first_model_test.png --save-report reports/first_model_test.json
```

## 3) STDP Training (Main STDP Pipeline)
Script: `STDP.py`

Features:
- LUT-STDP updates (pre/post traces)
- correlation monitor
- threshold-based neuron gating
- optional gated inference

Train:
```bash
python STDP.py train --train-dir preprocessed_train --test-dir preprocessed_test --enable-gating --corr-fan-in 16 --corr-threshold 0.9 --corr-min-steps 1000 --model-out checkpoints/snn_stdp.npz
```

Predict:
```bash
python STDP.py predict --model-path checkpoints/snn_stdp.npz --data-dir preprocessed_test --enable-gating
```

## 4) Full Architecture (Baseline vs Sparse)

### 4.1 Baseline architecture
Script: `full_architecture_snn.py`

```bash
python full_architecture_snn.py train --train-dir preprocessed_train --test-dir preprocessed_test --enable-gating --model-out checkpoints/full_arch_base.npz
python full_architecture_snn.py predict --model-path checkpoints/full_arch_base.npz --data-dir preprocessed_test --enable-gating
```

### 4.2 Sparse-index architecture (memory sparsity add-on)
Script: `full_architecture_snn_sparse.py`

Adds:
- indexed synapse fetch path (`pre_id -> nonzero (post_id, weight)`)
- periodic index rebuild
- optional small-weight pruning

Train:
```bash
python full_architecture_snn_sparse.py train --train-dir preprocessed_train --test-dir preprocessed_test --enable-gating --use-sparse-index --index-threshold 1e-4 --index-rebuild-interval 1 --prune-threshold 1e-4 --model-out checkpoints/full_arch_sparse.npz
```

Predict:
```bash
python full_architecture_snn_sparse.py predict --model-path checkpoints/full_arch_sparse.npz --data-dir preprocessed_test --enable-gating --use-sparse-index --index-threshold 1e-4 --index-rebuild-interval 1 --prune-threshold 1e-4
```

## 5) Memory and Weight Analysis

### 5.1 Training-time memory fetch analysis
Script: `train_memory_fetch_analysis.py`

```bash
python train_memory_fetch_analysis.py --train-dir preprocessed_train --epochs 2 --hidden-dim 512 --report-out reports/memory_fetch_report.json --save-sram
```

Reports counts for:
- SRAM reads/writes
- STDP update reads/writes
- correlation memory reads/writes
- trace reads/writes

### 5.2 Temporal and cumulative memory-read plot (testing)
Script: `plot_memory_reads_temporal.py`

```bash
python plot_memory_reads_temporal.py --model-path checkpoints/snn_stdp.npz --data-dir preprocessed_test --max-samples 100 --plot-out reports/temporal_memory_reads.png --report-out reports/temporal_memory_reads.json
```

Includes:
- per-time-bin reads
- cumulative reads over time
- burst and concentration stats

### 5.3 Weight change comparison
Script: `compare_weight_changes.py`

```bash
python compare_weight_changes.py --before <before_weights_or_model.npz> --after <after_weights_or_model.npz> --report-out reports/weight_change_report.json --plot-out reports/weight_change_report.png
```

Tip:
- If changed weights are `0`, you likely compared identical snapshots.

## 6) SRAM Utility
Script: `SRAM.py`

Provides SRAM-like weight memory:
- scalar/row/block read-write
- load/export array
- compressed save/load

## 7) Notes
- `SNN.py` is an experimental/demo path and not the main STDP architecture.
- For real classification performance, train on full `preprocessed_train` (all classes 0-9), not small smoke subsets.
>>>>>>> 5f6688c62 (Initial commit: N-MNIST preprocessing, STDP + correlation gating, sparse architecture, and analysis tools)
