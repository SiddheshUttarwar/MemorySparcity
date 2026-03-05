import os
import zipfile
import numpy as np
import matplotlib.pyplot as plt

def read_nmnist_events(data):
    events = np.frombuffer(data, dtype=np.uint8)
    events = events.reshape(-1, 5)
    x, y = events[:, 0], events[:, 1]
    p = (events[:, 2] > 0).astype(int)
    t = (events[:, 3].astype(np.uint32) << 8) | events[:, 4].astype(np.uint32)
    return x, y, p, t

def plot_spike_trains(raw_data_tuple, preprocessed_tensor, title_prefix, save_path):
    x, y, p, t = raw_data_tuple
    
    # 1. Raster Plot - Raw
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 2, 1)
    idx = y * 34 + x
    on_idx = p == 1
    off_idx = p == 0
    plt.scatter(t[on_idx], idx[on_idx], s=1, c='red', alpha=0.5, label='ON')
    plt.scatter(t[off_idx], idx[off_idx], s=1, c='blue', alpha=0.5, label='OFF')
    plt.title(f'{title_prefix} - Raw Raster (34x34)')
    plt.xlabel('Time (us)')
    plt.ylabel('Spatial ID (y*34 + x)')
    plt.legend()
    
    # 2. Raster Plot - Preprocessed
    # Since preprocessed is [T, 2, H, W] for native encoding
    # Let's extract spikes
    T, C, H, W = preprocessed_tensor.shape
    prep_t, prep_p, prep_y, prep_x = np.where(preprocessed_tensor == 1)
    
    plt.subplot(1, 2, 2)
    prep_idx = prep_y * W + prep_x
    p_on_idx = prep_p == 1
    p_off_idx = prep_p == 0
    
    plt.scatter(prep_t[p_on_idx], prep_idx[p_on_idx], s=1, c='red', alpha=0.5, label='ON')
    plt.scatter(prep_t[p_off_idx], prep_idx[p_off_idx], s=1, c='blue', alpha=0.5, label='OFF')
    plt.title(f'{title_prefix} - Preprocessed Raster ({W}x{H}, Binned Time)')
    plt.xlabel(f'Time Bins (0 to {T-1})')
    plt.ylabel(f'Spatial ID (y*{W} + x)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_temporal_stability(preprocessed_samples, digit, save_path):
    """
    Shows the sum over time to see the stability of the shape across multiple samples of the same digit.
    preprocessed_samples: list of [T, 2, H, W] tensors
    """
    plt.figure(figsize=(12, 4))
    for i, tensor in enumerate(preprocessed_samples[:4]): # Plot up to 4 samples
        plt.subplot(1, 4, i+1)
        # Sum over Time and Polarity
        summed = tensor.sum(axis=(0, 1))
        if summed.max() > 0:
            summed = summed / summed.max()
        plt.imshow(summed, cmap='hot')
        plt.title(f'Sample {i+1}')
        plt.axis('off')
    
    plt.suptitle(f'Temporal Stability (Accumulated Spikes) across {len(preprocessed_samples)} samples of Digit {digit}')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    prep_dir = 'preprocessed_data_native'
    zip_path = 'Test.zip'
    
    os.makedirs('plots_compare', exist_ok=True)
    
    # Let's grab the first few preprocessed test files for digits 0 and 1
    digit_samples = {0: [], 1: []}
    
    # Find preprocessed files for our target digits
    for file in os.listdir(prep_dir):
        if not file.startswith('test_'): continue
        if file.endswith('.npz'):
            parts = file.split('_')
            digit = int(parts[1])
            if digit in digit_samples and len(digit_samples[digit]) < 4:
                # Get corresponding original `.bin` file
                orig_name_base = parts[2].split('.npz')[0]
                orig_name = f"{orig_name_base}.bin"
                orig_zip_path = f"Test/{digit}/{orig_name}"
                digit_samples[digit].append((file, orig_zip_path))

    analysis_results = []
    analysis_results.append("# Data Comparison Results\n")
    
    with zipfile.ZipFile(zip_path, 'r') as z:
        for digit, samples in digit_samples.items():
            preprocessed_tensors = []
            
            analysis_results.append(f"## Digit {digit}\n")
            
            for i, (prep_file, orig_file) in enumerate(samples):
                # Load Preprocessed
                data = np.load(os.path.join(prep_dir, prep_file))
                prep_tensor = data['data'] # [T, 2, H, W] Boolean
                preprocessed_tensors.append(prep_tensor)
                
                # Load Raw
                raw_bytes = z.read(orig_file)
                raw_tuple = read_nmnist_events(raw_bytes)
                
                # 1. Event Rate/Density Analysis
                raw_events_count = len(raw_tuple[0])
                prep_events_count = np.sum(prep_tensor)
                
                # Density is events / total possible spatial/temporal locations
                # Raw spatial is 34x34, but time is continuous (just max_t - min_t)
                orig_t_span = max(1, raw_tuple[3].max() - raw_tuple[3].min())
                prep_t_span = prep_tensor.shape[0] # time bins
                prep_spatial = prep_tensor.shape[2] * prep_tensor.shape[3]
                
                # We show total raw events vs total spikes retained after cropping/binning/boolean-clipping
                reduction = (1 - (prep_events_count / max(1, raw_events_count))) * 100
                
                stats = (f"- **Sample {i+1} (`{orig_file}`)**:\n"
                         f"    - Raw Events: {raw_events_count}\n"
                         f"    - Preprocessed Spikes: {prep_events_count}\n"
                         f"    - Noise/Off-center Reduction: {reduction:.2f}%\n"
                         f"    - Note: Reduction comes from bounding box crop (34->28) and temporal boolean clipping inside bins.")
                analysis_results.append(stats)
                print(stats)
                
                # 2. Spike Train Visualization
                plot_spike_trains(raw_tuple, prep_tensor, f"Digit {digit} - Sample {i+1}", 
                                  os.path.join('plots_compare', f'digit_{digit}_sample_{i+1}_raster.png'))
            
            # 3. Temporal Stability
            plot_temporal_stability(preprocessed_tensors, digit, 
                                    os.path.join('plots_compare', f'digit_{digit}_stability.png'))
            analysis_results.append("\n")

    # Write textual summary
    with open('plots_compare/comparison_report.md', 'w') as f:
        f.write("\n".join(analysis_results))
        
    print("Comparison completed. Plots saved to 'plots_compare/' directory.")

if __name__ == "__main__":
    main()
