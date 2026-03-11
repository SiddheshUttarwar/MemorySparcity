---
marp: true
theme: default
class: lead
paginate: true
---

# Towards Zero-Waste Inference
### Hardware-Efficient Spiking Neural Networks via Dynamic Gatekeeping and Early Exit
**ECE274 Neuromorphic Computing Project**

---

# 1. Motivation: The Edge Computing Bottleneck

* **The Promise of SNNs:** Event-driven sparsity offers tremendous energy advantages for always-on edge sensors compared to standard dense ANNs.
* **The Reality of CMOS Hardware:** In digital silicon, fetching data from Memory (SRAM) costs $25\times$ more energy than the actual compute (MAC).
    * *SRAM Read (45nm):* $\sim 5$ pJ per access.
    * *INT8 Addition:* $\sim 0.2$ pJ per access.
* **The Problem:** Current SNN research focuses heavily on algorithmic accuracy but ignores the hardware cost of *unnecessary memory reads* caused by sensor noise and temporal redundancy.

---

# 2. Our Hardware-First Solution

We propose a natively digital, co-designed architecture that intercepts and drops redundant spikes *before* they can trigger expensive SRAM reads.

**Three Core Contributions:**
1. **Dynamic Gatekeeper Filtering:** Pre-processing input noise and burst redundancy.
2. **Adaptive Activity Regulation:** Hardware-efficient threshold adaptation to silence hidden-layer "spike storms".
3. **Early-Exit FSM:** Terminating the temporal processing window the moment the network reaches a confident prediction.

---

# 3. System-Level Architecture: Mult-Tile NoC

* To avoid a massive centralized memory bottleneck, our architecture is partitioned into **Tiles**. 
* Each layer (Conv1, Conv2, FC) operates as an independent hardware engine with strictly local, private SRAM.

![width:700px](file:///D:/Courses/ECE274-NeuromorphicComputing/Project/Hardware_Architecture/multi_tile_noc_diagram.png)

---

# 4. Inside the Datapath: The SNN Hardware Tile

Each Tile contains everything it needs to execute its layer:
* **Local SRAM Weight Bank:** Stores INT8 quantized weights.
* **Sparse MAC Array:** Adds weights conditionally based on binary spikes.
* **LIF Neuron Array:** Handles leaky integration and thresholding.

![width:600px](file:///D:/Courses/ECE274-NeuromorphicComputing/Project/Hardware_Architecture/snn_tile_diagram.png)

---

# 5. Core Innovation A: Dynamic Gatekeeper

Sits in front of the Conv1 tile to pre-filter the raw camera input.

* **Importance Monitor:** Filters *Spatial Noise*. Uses an array of 4-bit saturating counters with a global bit-shift decay. Drops isolated, random dark-current spikes.
* **Burst Redundancy Filter:** Filters *Temporal Noise*. A 12-bit register caches the last spike address. Drops consecutive, identical spikes caused by threshold jitter, saving redundant Conv1 SRAM reads.

![width:600px](file:///D:/Courses/ECE274-NeuromorphicComputing/Project/circuit_dynamic_gatekeeper.png)

---

# 6. Core Innovation B: Early-Exit FSM

Sits at the output classifier layer to truncate execution in the temporal dimension.

* **Concept:** Easy inputs (like a clear digit "1") don't need all 20 timesteps to be recognized.
* **Hardware:** 10 simple integer accumulators track class spikes. When any accumulator hits $\theta_{conf} = 8$, a global `Freeze` signal is raised.
* **Result:** The entire pipeline stops. No further timesteps are executed, saving 100% of the energy for the remaining temporal window.

---

# 7. Software Model & Training (PyTorch)

Hardware optimizations must be learned! We modeled the exact digital mechanics directly into a custom PyTorch LeNet-5 SNN.

* **Encoding:** N-MNIST DVS data is spatially cropped to $28 \times 28$ and temporally binned to $T=20$ discrete steps.
* **Learning:** Surrogate Gradient descent (STBP) with a cosine annealing schedule.
* **Co-Design:** The gatekeeper logic, adaptive thresholds, and early exit are active *during the forward pass*, forcing the weights to learn how to predict accurately under heavy hardware constraints.

---

# 8. Results: Massive SRAM Reduction

By combining spatial filtering and temporal truncation, average SRAM reads dropped by **81.6%** (from 70,824 to just 13,002 fetches per inference).

![width:800px](file:///D:/Courses/ECE274-NeuromorphicComputing/Project/fig1_cumulative_sram.png)

---

# 9. Results: Energy & Latency Scaling

Because SRAM dominates the power footprint, this 81.6% reduction in reads maps directly to an **81.6% reduction in estimated inference energy** ($\sim 68$ pJ).

![width:800px](file:///D:/Courses/ECE274-NeuromorphicComputing/Project/fig5_latency_speedup.png)

*Average latency also improved by $1.8\times$ due to Early Exit terminating inferences at $T=11.4$ on average!*

---

# 10. Conclusion

* **Hardware-First SNNs:** We proved that optimizing for algorithmic sparsity is not enough; we must optimize for *memory access*.
* **Zero-Waste Inference:** Simple, highly-efficient digital blocks (Ping-Pong buffers, 4-bit counters, shift-registers) can dramatically reduce the power footprint of neuromorphic chips.
* **Future Work:** Deploying the Verilog RTL onto an FPGA to gather physical synthesis, power, and timing reports.
