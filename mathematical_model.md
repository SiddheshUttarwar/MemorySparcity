# Mathematical Model — Sparse Neuromorphic SNN

This document formalizes all the mathematics behind the Sparse Convolutional Spiking Neural Network (CSNN), from raw event preprocessing through inference and hardware energy estimation.

---

## 1. N-MNIST Event Preprocessing

### 1.1 Raw Event Format
Each N-MNIST sample is a stream of asynchronous events from a neuromorphic camera:

$$e_i = (x_i,\ y_i,\ p_i,\ t_i)$$

Where:
- $(x_i, y_i) \in \{0, \ldots, 33\} \times \{0, \ldots, 33\}$ — pixel coordinate
- $p_i \in \{0, 1\}$ — polarity (OFF / ON)
- $t_i \in \mathbb{Z}^+$ — timestamp in microseconds

### 1.2 Spatial Cropping
The 34×34 sensor is center-cropped to 28×28 to match LeNet-5:

$$x'_i = x_i - 3, \quad y'_i = y_i - 3$$

Events outside the 28×28 region are discarded.

### 1.3 Temporal Binning
Continuous timestamps are discretized into $T = 20$ uniform time bins:

$$b_i = \text{clip}\!\left(\left\lfloor \frac{t_i - t_{\min}}{t_{\max} - t_{\min}} \cdot T \right\rfloor, \; 0, \; T-1 \right)$$

### 1.4 Spike Tensor Construction
Events are accumulated into a binary spike tensor $\mathbf{S} \in \{0,1\}^{T \times 2 \times 28 \times 28}$:

$$\mathbf{S}[b, p, y, x] = \begin{cases} 1 & \text{if any event } (x, y, p, t) \text{ falls in bin } b \\ 0 & \text{otherwise} \end{cases}$$

This is the input to the SNN: a 4D tensor with 2 polarity channels, 28×28 spatial, unrolled across $T=20$ time steps.

---

## 2. Leaky Integrate-and-Fire (LIF) Neuron

### 2.1 Membrane Potential Update
At each timestep $t$, the membrane potential of neuron $j$ in layer $\ell$ integrates the weighted input and decays:

$$v_j^{(\ell)}(t) = \beta \cdot v_j^{(\ell)}(t-1) + x_j^{(\ell)}(t)$$

Where:
- $\beta = 0.9$ — membrane leak factor (how much charge is retained between timesteps)
- $x_j^{(\ell)}(t)$ — input current (output of Conv/FC → BatchNorm)

### 2.2 Spike Generation (Heaviside)
A spike is emitted when membrane potential exceeds the threshold:

$$s_j^{(\ell)}(t) = \Theta\!\left(v_j^{(\ell)}(t) - v_{\text{th},j}^{(\ell)}(t)\right) = \begin{cases} 1 & \text{if } v_j^{(\ell)}(t) \geq v_{\text{th},j}^{(\ell)}(t) \\ 0 & \text{otherwise} \end{cases}$$

### 2.3 Soft Reset
After spiking, the membrane potential is reduced by the threshold (not reset to zero):

$$v_j^{(\ell)}(t) \leftarrow v_j^{(\ell)}(t) - s_j^{(\ell)}(t) \cdot v_{\text{th},j}^{(\ell)}(t)$$

This preserves residual charge above the threshold, maintaining temporal information.

### 2.4 Adaptive Threshold
The firing threshold increases after each spike and decays back to baseline:

$$v_{\text{th},j}^{(\ell)}(t+1) = v_{\text{th}}^{\text{base}} + 0.9 \cdot \left[\left(v_{\text{th},j}^{(\ell)}(t) + \rho \cdot s_j^{(\ell)}(t)\right) - v_{\text{th}}^{\text{base}}\right]$$

Where:
- $v_{\text{th}}^{\text{base}} = 1.0$ — base threshold
- $\rho = 0.1$ — adaptive scaling penalty per spike
- The $0.9\times$ decay factor slowly relaxes the elevated threshold back to baseline

**Effect:** Neurons that fire repeatedly face an increasingly high threshold, suppressing "spike storms" and enforcing temporal sparsity.

---

## 3. Surrogate Gradient (STBP)

The Heaviside function $\Theta(\cdot)$ has zero gradient almost everywhere, making backpropagation impossible. We replace its gradient with the **Fast Sigmoid** surrogate:

### 3.1 Forward Pass (unchanged)

$$s = \Theta(v - v_{\text{th}})$$

### 3.2 Backward Pass (surrogate)

$$\frac{\partial s}{\partial v} \approx \frac{1}{\left(1 + \alpha \cdot |v - v_{\text{th}}|\right)^2}$$

Where $\alpha = 2.0$ controls the sharpness of the surrogate. Higher $\alpha$ encourages sharper, more binary spike behavior.

---

## 4. Dynamic Gatekeeper

The Gatekeeper is an input-level filter that blocks noise and redundant spikes **before** they trigger SRAM reads. It combines two sub-filters:

### 4.1 Importance Monitor
A saturating counter $I_{c,y,x}(t)$ tracks the cumulative activity at each spatial position:

$$I_{c,y,x}(t) = I_{c,y,x}(t-1) + \mathbf{S}[t, c, y, x]$$

Every $W = 5$ timesteps, the counter is decayed via bit-shift:

$$I_{c,y,x}(t) \leftarrow \left\lfloor \frac{I_{c,y,x}(t)}{2} \right\rfloor \quad \text{when } t \bmod W = 0$$

An event is considered "important" if:

$$\text{imp\_keep}_{c,y,x}(t) = \begin{cases} \text{True} & \text{if } I_{c,y,x}(t) \geq \theta_{\text{imp}} \\ \text{False} & \text{otherwise} \end{cases}$$

Where $\theta_{\text{imp}} = 1.0$ (requires at least 1 prior spike at that location).

### 4.2 Burst Redundancy Filter
A repeat counter $R_{c,y,x}(t)$ tracks consecutive identical spikes:

$$R_{c,y,x}(t) = \begin{cases} R_{c,y,x}(t-1) + 1 & \text{if spike at } (c,y,x) \text{ at both } t \text{ and } t-1 \\ 0 & \text{otherwise} \end{cases}$$

An event passes the burst filter if:

$$\text{burst\_keep}_{c,y,x}(t) = \left(R_{c,y,x}(t) \leq R_{\max}\right), \quad R_{\max} = 1$$

### 4.3 Gate Decision
Only events passing **both** filters generate SRAM reads:

$$\text{gate}_{c,y,x}(t) = \text{is\_spike}(t) \;\wedge\; \text{imp\_keep}(t) \;\wedge\; \text{burst\_keep}(t)$$

$$\tilde{x}_{c,y,x}(t) = \text{gate}_{c,y,x}(t) \cdot \mathbf{S}[t, c, y, x]$$

**Hardware impact:** Each blocked spike saves one SRAM chip-select assertion and $K^2 \cdot C_{\text{out}}$ MAC operations.

---

## 5. Convolutional Layers & BatchNorm

### 5.1 Convolution
Standard 2D convolution with no bias:

$$x_j^{(\ell)}(t) = \sum_{i} \mathbf{W}_{ij}^{(\ell)} * s_i^{(\ell-1)}(t)$$

Where $*$ denotes convolution with kernel $\mathbf{W} \in \mathbb{R}^{C_{\text{out}} \times C_{\text{in}} \times K \times K}$.

### 5.2 Batch Normalization
Before the LIF neuron, we normalize to stabilize inputs near the firing threshold:

$$\hat{x}_j = \gamma_j \cdot \frac{x_j - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}} + \beta_j$$

Where $\gamma_j, \beta_j$ are learnable per-channel parameters. At inference, BN is folded into the convolution weights:

$$\mathbf{W}'_{ij} = \frac{\gamma_j}{\sqrt{\sigma^2_j + \epsilon}} \cdot \mathbf{W}_{ij}, \quad b'_j = \beta_j - \frac{\gamma_j \cdot \mu_j}{\sqrt{\sigma^2_j + \epsilon}}$$

### 5.3 Average Pooling
Reduces spatial dimensions by averaging $2 \times 2$ windows:

$$\text{Pool}(x)_{c,r,c'} = \frac{1}{4}\left(x_{c,2r,2c'} + x_{c,2r+1,2c'} + x_{c,2r,2c'+1} + x_{c,2r+1,2c'+1}\right)$$

We use AvgPool instead of MaxPool because LIF outputs are binary spikes ({0,1}). AvgPool preserves **firing density** (e.g., 0.25 = 1 of 4 neurons fired), while MaxPool only indicates if *any* neuron fired.

---

## 6. INT8 Symmetric Quantization

All weights are quantized to 8-bit signed integers for SRAM storage:

### 6.1 Quantization

$$s^{(\ell)} = \frac{127}{\max\!\left(|\mathbf{W}^{(\ell)}|\right)}$$

$$\hat{\mathbf{W}}^{(\ell)} = \text{clamp}\!\left(\left\lfloor s^{(\ell)} \cdot \mathbf{W}^{(\ell)} \right\rceil, \; -127, \; 127\right)$$

### 6.2 Dequantization (for hardware verification)

$$\mathbf{W}^{(\ell)}_{\text{approx}} = \frac{\hat{\mathbf{W}}^{(\ell)}}{s^{(\ell)}}$$

### 6.3 Storage
The quantized weights are stored in Verilog `.mem` hex files:

$$\text{hex}(w) = \text{format}\!\left(w \,\&\, \texttt{0xFF}, \; \texttt{'02x'}\right)$$

Negative values use two's complement: $w = -5 \rightarrow \text{0xFB}$.

---

## 7. Temporal Early Exit

### 7.1 Confidence Estimation
At each timestep $t \geq 3$, the cumulative output spike rate is computed:

$$\bar{r}_k(t) = \frac{1}{t+1} \sum_{\tau=0}^{t} s_k^{(\text{out})}(\tau), \quad k \in \{0, \ldots, 9\}$$

Then temperature-scaled softmax gives class probabilities:

$$p_k(t) = \frac{\exp\!\left(\tau \cdot \bar{r}_k(t)\right)}{\sum_{j=0}^{9} \exp\!\left(\tau \cdot \bar{r}_j(t)\right)}, \quad \tau = 5.0$$

### 7.2 Exit Criterion
The network halts if **all** samples in the batch are confident:

$$\text{HALT at } t^* = \min\!\left\{t \geq 3 \;\middle|\; \max_k p_k^{(b)}(t) > \theta_{\text{conf}} \;\; \forall b \in \text{batch}\right\}$$

Where $\theta_{\text{conf}} = 0.9$ (90% confidence margin).

### 7.3 Output Computation
The final classification uses the spike rate accumulated up to $t^*$:

$$\hat{y} = \arg\max_k \; \bar{r}_k(t^*)$$

---

## 8. Training Loss

### 8.1 Cross-Entropy Loss
Standard classification loss on the output spike rates:

$$\mathcal{L}_{\text{CE}} = -\sum_{k=0}^{9} y_k \cdot \log\!\left(\text{softmax}(\bar{r}_k)\right)$$

### 8.2 L1 Spike Regularization
Penalizes total spike activity across all layers and timesteps to encourage sparsity:

$$\mathcal{L}_{\text{spike}} = \lambda \cdot \sum_{t=0}^{T-1} \sum_{\ell} \sum_j s_j^{(\ell)}(t)$$

Where $\lambda = 5 \times 10^{-4}$ is the sparsity penalty weight.

### 8.3 Total Loss

$$\mathcal{L} = \mathcal{L}_{\text{CE}} + \mathcal{L}_{\text{spike}}$$

---

## 9. Hardware Energy Model

### 9.1 Per-Inference Energy
Total energy is dominated by SRAM access:

$$E_{\text{total}} = \sum_{t=0}^{t^*} \left[ N_{\text{reads}}(t) \cdot E_{\text{SRAM}} + N_{\text{MAC}}(t) \cdot E_{\text{MAC}} \right]$$

Where (at 45nm CMOS):
| Parameter | Value | Description |
|---|---|---|
| $E_{\text{SRAM}}$ | 5 pJ | Energy per 8-bit SRAM read (32KB bank) |
| $E_{\text{MAC}}$ | 0.2 pJ | Energy per 8-bit integer MAC |

### 9.2 SRAM Reads per Timestep
For each timestep $t$, the number of SRAM reads is:

$$N_{\text{reads}}(t) = \underbrace{N_{\text{gate}}(t)}_{\text{input layer}} + \underbrace{\sum_{\ell=1}^{L} S^{(\ell)}(t) \cdot F^{(\ell)}}_{\text{internal layers}}$$

Where:
- $N_{\text{gate}}(t)$ = spikes that passed the gatekeeper
- $S^{(\ell)}(t)$ = number of spikes in layer $\ell$ at time $t$
- $F^{(\ell)}$ = fan-out per spike in layer $\ell$ (e.g., $5 \times 5 \times 32 = 800$ for Conv1)

### 9.3 Savings Decomposition

$$\text{Gatekeeper savings} = 1 - \frac{\sum_t N_{\text{gate}}(t)}{\sum_t N_{\text{raw}}(t)} \approx 30\%$$

$$\text{Early exit savings} = 1 - \frac{t^*}{T} \approx 65\% \text{ (for easy digits)}$$

$$\text{Combined savings} = 1 - \frac{E_{\text{sparse}}}{E_{\text{baseline}}} \approx 60\text{-}70\%$$

### 9.4 Latency

$$L_{\text{baseline}} = T \cdot t_{\text{cycle}}, \quad L_{\text{sparse}} = t^* \cdot t_{\text{cycle}}$$

$$\text{Speedup} = \frac{T}{t^*}$$

Where $t_{\text{cycle}}$ is the clock period (e.g., 10 ns at 100 MHz).

---

## 10. Symbol Reference

| Symbol | Value | Description |
|---|---|---|
| $T$ | 20 | Maximum number of time bins |
| $\beta$ | 0.9 | Membrane leak factor |
| $v_{\text{th}}^{\text{base}}$ | 1.0 | Base firing threshold |
| $\rho$ | 0.1 | Adaptive threshold increment per spike |
| $\alpha$ | 2.0 | Surrogate gradient sharpness |
| $\tau$ | 5.0 | Softmax temperature for confidence |
| $\theta_{\text{conf}}$ | 0.9 | Early exit confidence threshold |
| $\lambda$ | $5 \times 10^{-4}$ | L1 spike regularization weight |
| $\theta_{\text{imp}}$ | 1.0 | Importance monitor threshold |
| $W$ | 5 | Importance counter decay window |
| $R_{\max}$ | 1 | Maximum allowed consecutive repeats |
| $E_{\text{SRAM}}$ | 5 pJ | SRAM read energy (45nm) |
| $E_{\text{MAC}}$ | 0.2 pJ | MAC operation energy (45nm) |
