import zipfile
import numpy as np
import matplotlib.pyplot as plt
import os

def read_nmnist_events(data):
    events = np.frombuffer(data, dtype=np.uint8)
    events = events.reshape(-1, 5)
    x = events[:, 0]
    y = events[:, 1]
    p = (events[:, 2] > 0).astype(int)  # 0 or 1
    t = (events[:, 3].astype(np.uint32) << 8) | events[:, 4].astype(np.uint32)
    return x, y, p, t

def plot_digit(sample_name, data, digit):
    x, y, p, t = read_nmnist_events(data)
    
    os.makedirs('plots', exist_ok=True)
    
    # Raster Plot
    plt.figure(figsize=(10, 4))
    idx = y * 34 + x
    on_idx = p == 1
    off_idx = p == 0
    plt.scatter(t[on_idx], idx[on_idx], s=1, c='red', alpha=0.5, label='ON (p=1)')
    plt.scatter(t[off_idx], idx[off_idx], s=1, c='blue', alpha=0.5, label='OFF (p=0)')
    plt.title(f'Raster Plot - Digit {digit}')
    plt.xlabel('Time')
    plt.ylabel('Flattened Spatial Index (y*34 + x)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'plots/digit_{digit}_raster.png')
    plt.close()

    # Spatial ON/OFF plot
    plt.figure(figsize=(5, 5))
    on_img = np.zeros((34, 34))
    off_img = np.zeros((34, 34))
    
    # accumulate events spatially
    np.add.at(on_img, (y[on_idx], x[on_idx]), 1)
    np.add.at(off_img, (y[off_idx], x[off_idx]), 1)
    
    # Plot as RGB image: Red for ON, Blue for OFF
    if on_img.max() > 0: on_img = on_img / on_img.max()
    if off_img.max() > 0: off_img = off_img / off_img.max()
    
    rgb = np.zeros((34, 34, 3))
    rgb[..., 0] = on_img  # Red
    rgb[..., 2] = off_img # Blue
    
    plt.imshow(rgb, interpolation='nearest')
    plt.title(f'Spatial ON/OFF Plot - Digit {digit}')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'plots/digit_{digit}_spatial.png')
    plt.close()

def main():
    with zipfile.ZipFile('Test.zip', 'r') as z:
        files = z.namelist()
        samples = {}
        for f in files:
            if not f.endswith('.bin'): continue
            parts = f.split('/')
            if len(parts) >= 3:
                digit = parts[1]
                if digit.isdigit() and digit not in samples:
                    samples[digit] = f
            if len(samples) == 10:
                break
        
        for d in range(10):
            d_str = str(d)
            if d_str in samples:
                with z.open(samples[d_str]) as f:
                    data = f.read()
                    plot_digit(samples[d_str], data, d_str)
                    print(f"Generated plots for digit {d_str}")

if __name__ == "__main__":
    main()
