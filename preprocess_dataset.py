import numpy as np
from scipy.ndimage import rotate
import os
import zipfile
import multiprocessing as mp
from functools import partial
import time
import math

class NeuromorphicPreprocessor:
    def __init__(self, target_size=(28, 28), original_size=(34, 34), time_bins=20):
        self.target_size = target_size
        self.original_size = original_size
        self.time_bins = time_bins

    def crop_events(self, x, y, p, t):
        offset_x = (self.original_size[0] - self.target_size[0]) // 2
        offset_y = (self.original_size[1] - self.target_size[1]) // 2
        
        mask = (x >= offset_x) & (x < offset_x + self.target_size[0]) & \
               (y >= offset_y) & (y < offset_y + self.target_size[1])
        return x[mask] - offset_x, y[mask] - offset_y, p[mask], t[mask]

    def native_event_to_tensor(self, x, y, p, t):
        spike_tensor = np.zeros((self.time_bins, 2, self.target_size[1], self.target_size[0]), dtype=np.bool_)
        if len(t) == 0:
            return spike_tensor
            
        t_min, t_max = t.min(), t.max()
        if t_max > t_min:
            t_norm = (t - t_min) / (t_max - t_min)
            t_binned = np.clip((t_norm * self.time_bins).astype(int), 0, self.time_bins - 1)
        else:
            t_binned = np.zeros_like(t, dtype=int)
            
        np.add.at(spike_tensor, (t_binned, p, y, x), True)
        return spike_tensor
        
    def one_hot_encode(self, label, num_classes=10):
        one_hot = np.zeros(num_classes, dtype=np.float32)
        one_hot[int(label)] = 1.0
        return one_hot

def read_nmnist_events(data):
    events = np.frombuffer(data, dtype=np.uint8)
    events = events.reshape(-1, 5)
    x, y = events[:, 0], events[:, 1]
    p = (events[:, 2] > 0).astype(int)
    t = (events[:, 3].astype(np.uint32) << 8) | events[:, 4].astype(np.uint32)
    return x, y, p, t

def process_file_chunk(file_infos_chunk, zip_path, out_dir):
    """ Processes a small chunk of zip entries to prevent multiprocessing hangs """
    processed = 0
    # Open local zipfile ref per worker
    with zipfile.ZipFile(zip_path, 'r') as z:
        processor = NeuromorphicPreprocessor(target_size=(28, 28), time_bins=20)
        for zip_info_name, dataset_name in file_infos_chunk:
            if not zip_info_name.endswith('.bin'):
                continue
                
            parts = zip_info_name.split('/')
            if len(parts) < 3:
                continue
                
            digit = int(parts[1])
            base_name = os.path.splitext(os.path.basename(zip_info_name))[0]
            out_file_path = os.path.join(out_dir, f"{dataset_name}_{digit}_{base_name}.npz")
            
            if os.path.exists(out_file_path):
                processed += 1
                continue # Already processed

            # Decode
            data = z.read(zip_info_name)
            x, y, p, t = read_nmnist_events(data)
            x_crop, y_crop, p_crop, t_crop = processor.crop_events(x, y, p, t)
            
            result_tensor = processor.native_event_to_tensor(x_crop, y_crop, p_crop, t_crop)
            label_one_hot = processor.one_hot_encode(digit)
            
            np.savez_compressed(out_file_path, data=result_tensor, label=label_one_hot, digit=digit)
            processed += 1
            
    return processed

def main():
    encoding_type = 'native'
    out_dir = f'preprocessed_data_{encoding_type}'
    os.makedirs(out_dir, exist_ok=True)
    
    datasets = [
        ('Test.zip', 'test'),
        ('Train.zip', 'train')
    ]
    
    processor_cores = max(1, mp.cpu_count() - 2)
    print(f"Starting FULL Dataset preprocessing natively with {processor_cores} cores. Saving to: {out_dir}")
    
    total_processed = 0
    start_time = time.time()
    
    for zip_path, dataset_name in datasets:
        if not os.path.exists(zip_path):
            print(f"WARNING: {zip_path} not found.")
            continue
            
        with zipfile.ZipFile(zip_path, 'r') as z:
            all_files = z.namelist()
            
        file_infos = [(f, dataset_name) for f in all_files if f.endswith('.bin')]
        print(f"Found {len(file_infos)} exact target files in {zip_path}")
        
        if not file_infos:
            continue
            
        # We partition the 60k/10k datasets into manageable chunks of 500 files each for the workers.
        chunk_size = 500
        num_chunks = math.ceil(len(file_infos) / chunk_size)
        chunks = [file_infos[i * chunk_size:(i + 1) * chunk_size] for i in range(num_chunks)]
        
        print(f"Divided {dataset_name} into {len(chunks)} processor chunks. Working...")

        worker_func = partial(process_file_chunk, zip_path=zip_path, out_dir=out_dir)
        
        processed_count = 0
        with mp.Pool(processes=processor_cores) as pool:
            for idx, c_res in enumerate(pool.imap_unordered(worker_func, chunks)):
                processed_count += c_res
                print(f" [{dataset_name}] Progress: {processed_count}/{len(file_infos)} files saved...")
            
        total_processed += processed_count
        print(f"Dataset {dataset_name} universally finished. Saved {processed_count} native .npz tensors.")
        
    print(f"\nAll completely done! Natively built {total_processed} files in {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    # Workaround for Windows Multiprocessing Fork Behavior
    mp.freeze_support()
    main()
