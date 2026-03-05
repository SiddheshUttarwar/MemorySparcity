import zipfile
import struct
import numpy as np

def analyze():
    with zipfile.ZipFile('Test.zip', 'r') as z:
        # Find one file per digit (0-9)
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
        
        # Read the first sample to determine format
        if not samples:
            print("No samples found.")
            return

        print("Samples found:")
        for k, v in samples.items():
            print(f"Digit {k}: {v}")

        sample_name = samples['0']
        with z.open(sample_name) as f:
            data = f.read()
            print(f"\nSize of {sample_name}: {len(data)} bytes")
            print("First 20 bytes (hex):", data[:20].hex())
            
            # Assuming N-MNIST format: each event is 5 bytes
            # x (1 byte), y (1 byte), p (1 byte), t (2 bytes)
            if len(data) % 5 == 0:
                print("Format fits 5-byte events.")
                events = np.frombuffer(data, dtype=np.uint8)
                events = events.reshape(-1, 5)
                print("First 5 events:")
                for i in range(min(5, len(events))):
                    x, y, p, t1, t2 = events[i]
                    print(f"x={x}, y={y}, p={p}, t1={t1}, t2={t2}")
            else:
                print("Format is NOT 5-byte events.")

if __name__ == "__main__":
    analyze()
