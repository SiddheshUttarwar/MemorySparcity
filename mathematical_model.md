# Mathematical Model — Sparse Neuromorphic SNN

This document formalizes all the mathematics behind the Sparse Convolutional Spiking Neural Network (CSNN), from raw event preprocessing through inference and hardware energy estimation.

---

## 1. N-MNIST Event Preprocessing

### 1.1 Raw Event Format
Each N-MNIST sample is a stream of asynchronous events from a neuromorphic camera:

```
event = (x, y, p, t)
  x, y ∈ {0, ..., 33}    pixel coordinate
  p    ∈ {0, 1}           polarity (OFF / ON)
  t    ∈ Z+               timestamp in microseconds
```

### 1.2 Spatial Cropping
The 34×34 sensor is center-cropped to 28×28 to match LeNet-5:

```
x' = x - 3
y' = y - 3
```

Events outside the 28×28 region are discarded.

### 1.3 Temporal Binning
Continuous timestamps are discretized into T = 20 uniform time bins:

```
bin(i) = clip( floor( (t_i - t_min) / (t_max - t_min) × T ),  0,  T-1 )
```

### 1.4 Spike Tensor Construction
Events are accumulated into a binary spike tensor S ∈ {0,1}^(T × 2 × 28 × 28):

```
S[b, p, y, x] = 1   if any event (x, y, p, t) falls in time bin b
               = 0   otherwise
```

This is the input to the SNN: 2 polarity channels, 28×28 spatial, unrolled across T=20 time steps.

---

## 2. Leaky Integrate-and-Fire (LIF) Neuron

### 2.1 Membrane Potential Update
At each timestep t, the membrane potential of neuron j in layer ℓ integrates input and decays:

```
v_j(t) = β × v_j(t-1) + x_j(t)
```

Where:
- `β = 0.9` — membrane leak factor (charge retained between timesteps)
- `x_j(t)` — input current (output of Conv/FC → BatchNorm)

### 2.2 Spike Generation (Heaviside)
A spike is emitted when membrane potential exceeds the threshold:

```
s_j(t) = Θ( v_j(t) - v_th_j(t) )

       = 1   if v_j(t) ≥ v_th_j(t)
       = 0   otherwise
```

### 2.3 Soft Reset
After spiking, the membrane potential is reduced by the threshold (not reset to zero):

```
v_j(t) ← v_j(t) - s_j(t) × v_th_j(t)
```

This preserves residual charge above the threshold, maintaining temporal information.

### 2.4 Adaptive Threshold
The firing threshold increases after each spike and decays back to baseline:

```
v_th_j(t+1) = v_th_base + 0.9 × [ (v_th_j(t) + ρ × s_j(t)) - v_th_base ]
```

Where:
- `v_th_base = 1.0` — base threshold
- `ρ = 0.1` — adaptive scaling penalty per spike
- `0.9×` decay factor slowly relaxes the elevated threshold back to baseline

**Effect:** Neurons that fire repeatedly face an increasingly high threshold, suppressing "spike storms" and enforcing temporal sparsity.

---

## 3. Surrogate Gradient (STBP)

The Heaviside function Θ(·) has zero gradient almost everywhere, making backpropagation impossible. We replace its gradient with the **Fast Sigmoid** surrogate:

### 3.1 Forward Pass (unchanged)

```
s = Θ(v - v_th)     (standard Heaviside — 0 or 1)
```

### 3.2 Backward Pass (surrogate gradient)

```
∂s/∂v  ≈  1 / (1 + α × |v - v_th|)²
```

Where `α = 2.0` controls sharpness. Higher α encourages sharper, more binary spike behavior.

---

## 4. Dynamic Gatekeeper

The Gatekeeper is an input-level filter that blocks noise and redundant spikes **before** they trigger SRAM reads. It combines two sub-filters:

### 4.1 Importance Monitor
A saturating counter I(c,y,x,t) tracks cumulative activity at each spatial position:

```
I(c,y,x,t) = I(c,y,x,t-1) + S[t, c, y, x]
```

Every W = 5 timesteps, the counter is decayed via bit-shift (hardware-friendly divide-by-2):

```
I(c,y,x,t) ← floor( I(c,y,x,t) / 2 )    when t mod W = 0
```

An event is considered "important" if:

```
imp_keep(c,y,x,t) = True    if I(c,y,x,t) ≥ θ_imp
                   = False   otherwise

θ_imp = 1.0   (requires at least 1 prior spike at that location)
```

### 4.2 Burst Redundancy Filter
A repeat counter R(c,y,x,t) tracks consecutive identical spikes:

```
R(c,y,x,t) = R(c,y,x,t-1) + 1    if spike at (c,y,x) at BOTH t and t-1
            = 0                     otherwise
```

An event passes the burst filter if:

```
burst_keep(c,y,x,t) = ( R(c,y,x,t) ≤ R_max )

R_max = 1   (allow at most 1 consecutive repeat)
```

### 4.3 Gate Decision
Only events passing **both** filters generate SRAM reads:

```
gate(c,y,x,t) = is_spike(t)  AND  imp_keep(t)  AND  burst_keep(t)

x_filtered(c,y,x,t) = gate(c,y,x,t) × S[t, c, y, x]
```

**Hardware impact:** Each blocked spike saves one SRAM chip-select assertion and K² × C_out MAC operations.

---

## 5. Convolutional Layers & BatchNorm

### 5.1 Convolution
Standard 2D convolution with no bias:

```
x_j(t) = Σ_i  W_ij * s_i(t)
```

Where `*` denotes convolution with kernel W ∈ R^(C_out × C_in × K × K).

### 5.2 Batch Normalization
Before the LIF neuron, we normalize to stabilize inputs near the firing threshold:

```
x_hat_j = γ_j × (x_j - μ_B) / sqrt(σ²_B + ε)  +  β_j
```

Where γ, β are learnable per-channel parameters. At inference, BN is folded into conv weights:

```
W'_ij = (γ_j / sqrt(σ²_j + ε)) × W_ij

b'_j  = β_j - (γ_j × μ_j) / sqrt(σ²_j + ε)
```

### 5.3 Average Pooling
Reduces spatial dimensions by averaging 2×2 windows:

```
Pool(x)_(c,r,c') = (1/4) × [ x_(c,2r,2c') + x_(c,2r+1,2c') + x_(c,2r,2c'+1) + x_(c,2r+1,2c'+1) ]
```

We use AvgPool instead of MaxPool because LIF outputs are binary spikes {0,1}. AvgPool preserves **firing density** (e.g., 0.25 = 1 of 4 neurons fired), while MaxPool only indicates if any neuron fired.

---

## 6. INT8 Symmetric Quantization

All weights are quantized to 8-bit signed integers for SRAM storage:

### 6.1 Quantization

```
scale = 127 / max(|W|)

W_q = clamp( round(scale × W),  -127,  127 )
```

### 6.2 Dequantization (for hardware verification)

```
W_approx = W_q / scale
```

### 6.3 Storage Format
Quantized weights are stored in Verilog `.mem` hex files using two's complement:

```
hex(w) = format(w & 0xFF, '02x')

Example: w = -5  →  0xFB  (two's complement)
         w = 42  →  0x2A
```

---

## 7. Temporal Early Exit

### 7.1 Confidence Estimation
At each timestep t ≥ 3, the cumulative output spike rate is computed:

```
r_k(t) = (1 / (t+1)) × Σ_{τ=0}^{t}  s_k_out(τ)     for k ∈ {0, ..., 9}
```

Then temperature-scaled softmax gives class probabilities:

```
p_k(t) = exp(τ × r_k(t)) / Σ_j exp(τ × r_j(t))

τ = 5.0   (temperature scaling factor)
```

### 7.2 Exit Criterion
The network halts if **all** samples in the batch are confident:

```
HALT at t* = min{ t ≥ 3  |  max_k p_k(b,t) > θ_conf   for all b in batch }

θ_conf = 0.9   (90% confidence margin)
```

### 7.3 Output Computation
Final classification uses the spike rate accumulated up to t*:

```
y_hat = argmax_k  r_k(t*)
```

---

## 8. Training Loss

### 8.1 Cross-Entropy Loss
Standard classification loss on the output spike rates:

```
L_CE = -Σ_k  y_k × log( softmax(r_k) )
```

### 8.2 L1 Spike Regularization
Penalizes total spike activity across all layers and timesteps to encourage sparsity:

```
L_spike = λ × Σ_{t=0}^{T-1}  Σ_ℓ  Σ_j  s_j(ℓ,t)

λ = 5 × 10⁻⁴   (sparsity penalty weight)
```

### 8.3 Total Loss

```
L_total = L_CE + L_spike
```

---

## 9. Hardware Energy Model

### 9.1 Per-Inference Energy
Total energy is dominated by SRAM access:

```
E_total = Σ_{t=0}^{t*}  [ N_reads(t) × E_SRAM  +  N_MAC(t) × E_MAC ]
```

Where (at 45nm CMOS):

| Parameter | Value | Description |
|-----------|-------|-------------|
| E_SRAM | 5 pJ | Energy per 8-bit SRAM read (32KB bank) |
| E_MAC | 0.2 pJ | Energy per 8-bit integer MAC |

### 9.2 SRAM Reads per Timestep

```
N_reads(t) = N_gate(t)  +  Σ_ℓ  S_ℓ(t) × F_ℓ
```

Where:
- `N_gate(t)` = spikes that passed the gatekeeper (input layer reads)
- `S_ℓ(t)` = number of spikes in layer ℓ at time t
- `F_ℓ` = fan-out per spike (e.g., 5×5×32 = 800 for Conv1)

### 9.3 Savings Decomposition

```
Gatekeeper savings  = 1 - Σ_t N_gate(t) / Σ_t N_raw(t)          ≈ 30%

Early exit savings  = 1 - t* / T                                  ≈ 65% (easy digits)

Combined savings    = 1 - E_sparse / E_baseline                   ≈ 60-70%
```

### 9.4 Latency

```
L_baseline = T × t_cycle
L_sparse   = t* × t_cycle

Speedup = T / t*
```

Where `t_cycle` is the clock period (e.g., 10 ns at 100 MHz).

---

## 10. Symbol Reference

| Symbol | Value | Description |
|--------|-------|-------------|
| T | 20 | Maximum number of time bins |
| β | 0.9 | Membrane leak factor |
| v_th_base | 1.0 | Base firing threshold |
| ρ | 0.1 | Adaptive threshold increment per spike |
| α | 2.0 | Surrogate gradient sharpness |
| τ | 5.0 | Softmax temperature for confidence |
| θ_conf | 0.9 | Early exit confidence threshold |
| λ | 5 × 10⁻⁴ | L1 spike regularization weight |
| θ_imp | 1.0 | Importance monitor threshold |
| W | 5 | Importance counter decay window |
| R_max | 1 | Maximum allowed consecutive repeats |
| E_SRAM | 5 pJ | SRAM read energy (45nm) |
| E_MAC | 0.2 pJ | MAC operation energy (45nm) |
