from pathlib import Path

import numpy as np


class SRAMWeightMemory:
    """
    SRAM-like memory model to store connection weights.

    Memory is represented as a 2D array: [rows, cols].
    Typical mapping:
    - rows: source neurons
    - cols: destination neurons
    """

    def __init__(self, rows, cols, dtype=np.float32, init="zeros", seed=0):
        self.rows = int(rows)
        self.cols = int(cols)
        self.dtype = dtype
        self.mem = np.zeros((self.rows, self.cols), dtype=self.dtype)

        if init == "zeros":
            pass
        elif init == "xavier":
            rng = np.random.default_rng(seed)
            scale = np.sqrt(2.0 / max(1, self.rows + self.cols))
            self.mem = rng.normal(0.0, scale, size=(self.rows, self.cols)).astype(self.dtype)
        elif init == "uniform":
            rng = np.random.default_rng(seed)
            self.mem = rng.uniform(-0.1, 0.1, size=(self.rows, self.cols)).astype(self.dtype)
        else:
            raise ValueError(f"Unsupported init mode: {init}")

    def shape(self):
        return self.mem.shape

    def read(self, row, col):
        return self.mem[int(row), int(col)]

    def write(self, row, col, value):
        self.mem[int(row), int(col)] = self.dtype(value)

    def read_row(self, row):
        return self.mem[int(row), :].copy()

    def write_row(self, row, values):
        values = np.asarray(values, dtype=self.dtype)
        if values.shape[0] != self.cols:
            raise ValueError(f"Row length mismatch: expected {self.cols}, got {values.shape[0]}")
        self.mem[int(row), :] = values

    def read_block(self, row_start, row_end, col_start, col_end):
        return self.mem[int(row_start) : int(row_end), int(col_start) : int(col_end)].copy()

    def write_block(self, row_start, row_end, col_start, col_end, values):
        values = np.asarray(values, dtype=self.dtype)
        target = self.mem[int(row_start) : int(row_end), int(col_start) : int(col_end)]
        if target.shape != values.shape:
            raise ValueError(f"Block shape mismatch: expected {target.shape}, got {values.shape}")
        target[:, :] = values

    def load_from_array(self, weights):
        w = np.asarray(weights, dtype=self.dtype)
        if w.shape != (self.rows, self.cols):
            raise ValueError(f"Weight shape mismatch: expected {(self.rows, self.cols)}, got {w.shape}")
        self.mem[:, :] = w

    def export_array(self):
        return self.mem.copy()

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, weights=self.mem)

    @classmethod
    def load(cls, path, dtype=np.float32):
        d = np.load(path)
        w = d["weights"].astype(dtype)
        obj = cls(rows=w.shape[0], cols=w.shape[1], dtype=dtype, init="zeros")
        obj.load_from_array(w)
        return obj


if __name__ == "__main__":
    # Minimal smoke example
    sram = SRAMWeightMemory(rows=4, cols=3, init="xavier", seed=42)
    print("Shape:", sram.shape())
    print("w[0,0] before:", float(sram.read(0, 0)))
    sram.write(0, 0, 0.75)
    print("w[0,0] after :", float(sram.read(0, 0)))
    sram.save("checkpoints/sram_weights_demo.npz")
