# LeNet-5 Sparse CSNN: Architecture & Shape Parameters

This document breaks down the structural details, layer shapes, and parameter counts for the Spiking Convolutional Neural Network (CSNN) used in this project. 

The network processes the natively event-based N-MNIST dataset over **$T=20$ time bins**. The table below represents the spatial transformations that occur during a **single timestep**.

## Layer-by-Layer Shape Transformations

| Layer Name | Type | Input Shape (C, H, W) | Output Shape (C, H, W) | Kernel (H, W) | Stride | Pad | Parameters |
|:---|:---|:---|:---|:---:|:---:|:---:|---:|
| **Input** | N-MNIST Event Frame | `(2, 28, 28)` | — | — | — | — | 0 |
| `conv1` | `Conv2d` | `(2, 28, 28)` | `(32, 28, 28)` | 5 × 5 | 1 | 2 | 1,600 |
| `bn1` | `BatchNorm2d` | `(32, 28, 28)` | `(32, 28, 28)` | — | — | — | 64 |
| `lif1` | `Adaptive LIF` | `(32, 28, 28)` | `(32, 28, 28)` | — | — | — | 0 |
| `pool1` | `AvgPool2d` | `(32, 28, 28)` | `(32, 14, 14)` | 2 × 2 | 2 | 0 | 0 |
| `conv2` | `Conv2d` | `(32, 14, 14)` | `(64, 14, 14)` | 5 × 5 | 1 | 2 | 51,200 |
| `bn2` | `BatchNorm2d` | `(64, 14, 14)` | `(64, 14, 14)` | — | — | — | 128 |
| `lif2` | `Adaptive LIF` | `(64, 14, 14)` | `(64, 14, 14)` | — | — | — | 0 |
| `pool2` | `AvgPool2d` | `(64, 14, 14)` | `(64, 7, 7)` | 2 × 2 | 2 | 0 | 0 |
| `flatten` | `Flatten` | `(64, 7, 7)` | `(3136)` (1D) | — | — | — | 0 |
| `dropout` | `Dropout(0.5)` | `(3136)` | `(3136)` | — | — | — | 0 |
| `fc1` | `Linear` | `(3136)` | `(128)` | — | — | — | 401,408 |
| `lif3` | `Adaptive LIF` | `(128)` | `(128)` | — | — | — | 0 |
| `fc2` | `Linear` | `(128)` | `(10)` | — | — | — | 1,280 |
| `lif4` | `Adaptive LIF` | `(10)` | `(10)` | — | — | — | 0 |

---

### Summary
* **Total Trainable Parameters:** **455,680**
* *Note:* All `Conv2d` and `Linear` layers are configured to be computationally bias-free (`bias=False`), optimizing the hardware footprint by reserving parameter memory exclusively for the quantized INT8 weights. The time dimension ($T$) behaves recursively, unrolling this entire block across time steps rather than multiplying the parameter count.
