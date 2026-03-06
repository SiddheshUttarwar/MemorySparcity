import torch
import random
from train import NMNISTDataset
from sparse_snn_model import LeNet5_Sparse_CSNN

def predict_single_samples():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "="*50)
    print(f"LOADING SPARSE SNN INFERENCE ENGINE ({DEVICE})")
    print("="*50)

    # 1. Initialize the Hardware Model
    model = LeNet5_Sparse_CSNN(in_channels=2, num_classes=10).to(DEVICE)
    
    # 2. Load the best physical weights from the Colab training
    try:
        model.load_state_dict(torch.load("best_sparse_model.pth", map_location=DEVICE))
        print("✅ Successfully loaded weights from 'best_sparse_model.pth'")
    except FileNotFoundError:
        print("❌ Error: 'best_sparse_model.pth' not found.")
        print("Did you finish running train_sparse.py on Colab? Double check the file exists.")
        return

    # Put Model in Testing Mode (Disables Dropout and enforces Inference physics)
    model.eval()
    
    # Pre-sync the PyTorch weights into our simulated Quantized SRAM banks
    model.sync_to_sram()
    model.sync_from_sram()

    # 3. Load Testing Data
    test_dataset = NMNISTDataset(data_dir='preprocessed_data_native', split='test')
    
    # 4. Grab 5 Random Samples
    num_samples = 5
    indices = random.sample(range(len(test_dataset)), num_samples)

    print("\n" + "="*50)
    print(f"EXECUTING INFERENCES ({num_samples} SAMPLES)")
    print("="*50)

    total_accuracy = 0

    # 5. Execute Prediction and Fetch Physical Hardware Stats
    for i, idx in enumerate(indices):
        x_seq, label = test_dataset[idx]
        
        # Add Batch Dimension [1, T, C, H, W]
        x_seq = x_seq.unsqueeze(0).to(DEVICE)  

        with torch.no_grad():
            # Run the Spiking Neural Network Forward Pass!
            spike_rate, output_spikes, l1_spike_sum, actual_steps, hw_metrics = model(x_seq, early_exit=True)

        _, predicted = spike_rate.max(1)
        pred_label = predicted.item()

        # Did it get it right?
        correct = (label == pred_label)
        if correct: total_accuracy += 1

        # ========================================================
        # CALCULATING ABSOLUTE SRAM MEMORY READS
        # ========================================================
        # 1. Input Layer Memory Reads
        # (How many times did the Gatekeeper allow an incoming Spike to ping the Conv1 SRAM?)
        raw_input_spikes = hw_metrics['total_in']
        kept_input_spikes = hw_metrics['kept_in']
        gate_block_pct = 100 * (1 - (kept_input_spikes / max(1, raw_input_spikes)))
        
        # 2. Internal Network Memory Reads
        # (Total spikes generated internally across all Convolutional and Dense layers)
        # Every internal spike triggers an SRAM read in the *next* layer's physical memory bank.
        internal_spikes = int(l1_spike_sum.item())
        
        # 3. Total Memory Fetches
        total_sram_reads = kept_input_spikes + internal_spikes

        # Console Output
        print(f"\n[Sample {i+1}/{num_samples}] -> True Label: {label} | Predicted: {pred_label}")
        print(f"  Result           : {'✅ CORRECT' if correct else '❌ INCORRECT'}")
        print(f"  Time Steps Used  : {actual_steps} / 20  (Early-Exit Confidence Reached)")
        print(f"  Sensory Noise    : {raw_input_spikes:,} raw spikes")
        print(f"  SRAM Reads (Input) : {kept_input_spikes:,}  (Gatekeeper Blocked {gate_block_pct:.1f}%)")
        print(f"  SRAM Reads (Hidden): {internal_spikes:,}  (Adaptive Thresholds applied)")
        print(f"  ------------------------------------------------")
        print(f"  TOTAL SRAM READS = {total_sram_reads:,}")

    print("\n" + "="*50)
    print(f"Inference Batch Complete! Local Accuracy: {total_accuracy}/{num_samples} ({(total_accuracy/num_samples)*100:.1f}%)")

if __name__ == "__main__":
    predict_single_samples()
