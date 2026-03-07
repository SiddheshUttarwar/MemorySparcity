import torch
import torch.nn as nn
import math
from SRAM import SRAMWeightMemory
from gatekeeper import GatekeeperController

class SurrogateFastSigmoid(torch.autograd.Function):
    """
    Surrogate Gradient for the non-differentiable Heaviside step function.
    Uses a stepper alpha=2.0 for sparse SNN to encourage stronger binary behavior without exploding gradients.
    """
    alpha = 2.0 

    @staticmethod
    def forward(ctx, input_tensor, threshold):
        ctx.save_for_backward(input_tensor, threshold)
        out = torch.zeros_like(input_tensor)
        out[input_tensor >= threshold] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, threshold = ctx.saved_tensors
        grad_input = grad_output.clone()
        surrogate_grad = 1.0 / (1.0 + SurrogateFastSigmoid.alpha * torch.abs(input_tensor - threshold)).pow(2)
        return grad_input * surrogate_grad, None

class LIFNodeSTBP_Sparse(nn.Module):
    """
    Adaptive LIF Neuron that dynamically raises its threshold if it spikes, 
    suppressing "spike storms" across the simulated SRAM bus.
    """
    def __init__(self, beta=0.9, v_threshold=1.5, v_rest=0.0, rho=0.1):
        super().__init__()
        self.beta = beta
        self.base_v_th = v_threshold
        self.v_rest = v_rest
        # rho: adaptive threshold scaling penalty
        self.rho = rho 

    def forward(self, x_t, v, v_th):
        # 1. Integrate
        v = v * self.beta + x_t
        # 2. Spike
        spike = SurrogateFastSigmoid.apply(v, v_th)
        # 3. Soft Reset
        v = v - spike * v_th
        # 4. Adaptive Thresholding (v_th increases when spiking)
        v_th = v_th + self.rho * spike 
        # Decay threshold softly back to base over time
        v_th = self.base_v_th + 0.9 * (v_th - self.base_v_th) 
        
        return spike, v, v_th

class LeNet5_Sparse_CSNN(nn.Module):
    def __init__(self, in_channels=2, num_classes=10, config=None):
        super().__init__()
        # Load config or use defaults
        self.use_gatekeeper = config.use_gatekeeper if config else True
        self.use_early_exit = config.use_early_exit if config else True
        self.use_adaptive_threshold = config.use_adaptive_threshold if config else True
        self.use_sparsity_reg = config.use_sparsity_reg if config else True
        self.confidence_margin = config.confidence_margin if config else 0.9
        self.temperature = config.temperature if config else 5.0

        self.beta = config.beta if config else 0.9
        self.v_th = config.v_threshold if config else 1.0
        self.rho = config.rho if config else 0.05

        # Effective rho: 0 when adaptive threshold is disabled
        effective_rho = self.rho if self.use_adaptive_threshold else 0.0

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=5, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.lif1 = LIFNodeSTBP_Sparse(beta=self.beta, v_threshold=self.v_th, rho=effective_rho)
        self.pool1 = nn.AvgPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.lif2 = LIFNodeSTBP_Sparse(beta=self.beta, v_threshold=self.v_th, rho=effective_rho)
        self.pool2 = nn.AvgPool2d(2, 2)

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(config.dropout if config else 0.5)

        self.fc1 = nn.Linear(64 * 7 * 7, 128, bias=False)
        self.lif3 = LIFNodeSTBP_Sparse(beta=self.beta, v_threshold=self.v_th, rho=effective_rho)

        self.fc2 = nn.Linear(128, num_classes, bias=False)
        self.lif4 = LIFNodeSTBP_Sparse(beta=self.beta, v_threshold=self.v_th, rho=effective_rho)

        # Formalized Gatekeeper Controller
        self.gatekeeper = GatekeeperController(
            channels=in_channels, height=28, width=28,
            imp_thresh=config.imp_thresh if config else 1.0,
            win_tick=config.imp_win_tick if config else 5,
            max_repeats=config.max_repeats if config else 1,
            enabled=self.use_gatekeeper,
        )

        self._initialize_weights()
        self._init_sram()

    def _init_sram(self):
        self.sram_memories = {}
        self.sram_memories['conv1'] = SRAMWeightMemory(rows=32, cols=2*5*5)
        self.sram_memories['conv2'] = SRAMWeightMemory(rows=64, cols=32*5*5)
        self.sram_memories['fc1'] = SRAMWeightMemory(rows=128, cols=64*7*7)
        self.sram_memories['fc2'] = SRAMWeightMemory(rows=10, cols=128)
        self.sync_to_sram()

    def sync_to_sram(self):
        def quantize_int8(w):
            """Simulates 8-bit SRAM quantization clamping."""
            w_max = w.abs().max()
            if w_max < 1e-6:
                return w.detach().cpu().numpy(), 1.0
            scale = w_max / 127.0
            q_w = torch.round(w / scale) * scale
            return q_w.detach().cpu().numpy(), scale.item()

        # Save quantized weights 
        c1_w, _ = quantize_int8(self.conv1.weight)
        self.sram_memories['conv1'].load_from_array(c1_w.reshape(32, -1))
        
        c2_w, _ = quantize_int8(self.conv2.weight)
        self.sram_memories['conv2'].load_from_array(c2_w.reshape(64, -1))
        
        fc1_w, _ = quantize_int8(self.fc1.weight)
        self.sram_memories['fc1'].load_from_array(fc1_w)
        
        fc2_w, _ = quantize_int8(self.fc2.weight)
        self.sram_memories['fc2'].load_from_array(fc2_w)

    def sync_from_sram(self):
        device = self.conv1.weight.device
        
        c1_w = self.sram_memories['conv1'].export_array().reshape(32, 2, 5, 5)
        self.conv1.weight.data.copy_(torch.tensor(c1_w, device=device, dtype=torch.float32))
        
        c2_w = self.sram_memories['conv2'].export_array().reshape(64, 32, 5, 5)
        self.conv2.weight.data.copy_(torch.tensor(c2_w, device=device, dtype=torch.float32))
        
        fc1_w = self.sram_memories['fc1'].export_array()
        self.fc1.weight.data.copy_(torch.tensor(fc1_w, device=device, dtype=torch.float32))
        
        fc2_w = self.sram_memories['fc2'].export_array()
        self.fc2.weight.data.copy_(torch.tensor(fc2_w, device=device, dtype=torch.float32))

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x_seq, early_exit=None, confidence_margin=None):
        B, T, C, H, W = x_seq.shape
        device = x_seq.device

        # Use instance defaults if not overridden
        if early_exit is None:
            early_exit = self.use_early_exit
        if confidence_margin is None:
            confidence_margin = self.confidence_margin

        # Initialize membrane potentials and dynamic thresholds
        v1 = torch.zeros(B, 32, H, W, device=device)
        v_th1 = torch.full((B, 32, H, W), self.v_th, device=device)

        v2 = torch.zeros(B, 64, H//2, W//2, device=device)
        v_th2 = torch.full((B, 64, H//2, W//2), self.v_th, device=device)

        v3 = torch.zeros(B, 128, device=device)
        v_th3 = torch.full((B, 128), self.v_th, device=device)

        v4 = torch.zeros(B, 10, device=device)
        v_th4 = torch.full((B, 10), self.v_th, device=device)

        # Initialize Gatekeeper state for this batch
        self.gatekeeper.reset_state(B, device)

        # Hardware Counters
        hw_cs_asserts = 0
        hw_mac_ops = 0
        hw_total_raw_in = 0
        hw_kept_in = 0

        out_spikes_seq = []
        l1_spike_sum = torch.tensor(0.0, device=device, requires_grad=True)

        cumulative_sram_reads = []
        current_cumulative_reads = 0

        # Per-timestep spike tracking
        per_layer_spikes = {0: 0, 1: 0, 2: 0, 3: 0}
        active_neurons_per_t = []
        confidence_trajectory = []

        actual_time_steps = T

        # Step-wise unrolling to allow Temporal Early-Exit
        for t in range(T):
            x_raw_t = x_seq[:, t, ...]
            hw_total_raw_in += (x_raw_t > 0).sum().item()

            # --- Gatekeeper Decision (formalized module) ---
            x_t, gate_metrics = self.gatekeeper(x_raw_t, t)

            active_spikes = gate_metrics.kept
            hw_kept_in += active_spikes
            hw_cs_asserts += active_spikes
            hw_mac_ops += active_spikes * (5 * 5 * 32)

            # Layer Pass
            x_t = self.conv1(x_t)
            x_t = self.bn1(x_t)
            spike1, v1, v_th1 = self.lif1(x_t, v1, v_th1)
            x_t = self.pool1(spike1)

            x_t = self.conv2(x_t)
            x_t = self.bn2(x_t)
            spike2, v2, v_th2 = self.lif2(x_t, v2, v_th2)
            x_t = self.pool2(spike2)

            x_t = self.flatten(x_t)
            x_t = self.dropout(x_t)

            x_t = self.fc1(x_t)
            spike3, v3, v_th3 = self.lif3(x_t, v3, v_th3)

            x_t = self.fc2(spike3)
            out_spike, v4, v_th4 = self.lif4(x_t, v4, v_th4)

            out_spikes_seq.append(out_spike)

            # Per-layer spike counting
            s1 = spike1.sum()
            s2 = spike2.sum()
            s3 = spike3.sum()
            s4 = out_spike.sum()
            per_layer_spikes[0] += int(s1.item())
            per_layer_spikes[1] += int(s2.item())
            per_layer_spikes[2] += int(s3.item())
            per_layer_spikes[3] += int(s4.item())

            step_internal_spikes = s1 + s2 + s3 + s4
            l1_spike_sum = l1_spike_sum + step_internal_spikes

            # Active neurons this timestep
            active_neurons = int((spike1 > 0).sum().item() + (spike2 > 0).sum().item() +
                                 (spike3 > 0).sum().item() + (out_spike > 0).sum().item())
            active_neurons_per_t.append(active_neurons)

            # Cumulative SRAM reads
            current_cumulative_reads += active_spikes + int(step_internal_spikes.item())
            cumulative_sram_reads.append(current_cumulative_reads)

            # Confidence tracking & early exit
            current_spikes = torch.stack(out_spikes_seq, dim=1)
            current_rate = current_spikes.mean(dim=1)
            probs = torch.softmax(current_rate * self.temperature, dim=1)
            max_probs, _ = probs.max(dim=1)
            confidence_trajectory.append(float(max_probs.mean().item()))

            if early_exit and t >= 3:
                if (max_probs > confidence_margin).all():
                    actual_time_steps = t + 1
                    break

        # Pad sequence with zeros if early exit happened
        while len(out_spikes_seq) < T:
            out_spikes_seq.append(torch.zeros_like(out_spikes_seq[-1]))

        out_spikes = torch.stack(out_spikes_seq, dim=1)
        spike_rate = out_spikes[:, :actual_time_steps, :].mean(dim=1)

        # Structured hardware metrics (backward compatible + extended)
        gk_summary = self.gatekeeper.get_summary()
        hw_metrics = {
            'cs_asserts': hw_cs_asserts,
            'mac_ops': hw_mac_ops,
            'total_in': hw_total_raw_in,
            'kept_in': hw_kept_in,
            'cumulative_reads_over_time': cumulative_sram_reads,
            # Extended metrics for publication
            'per_layer_spikes': per_layer_spikes,
            'active_neurons_per_t': active_neurons_per_t,
            'confidence_trajectory': confidence_trajectory,
            'gk_summary': gk_summary,
            'sram_reads_hidden': int(l1_spike_sum.item()),
        }

        return spike_rate, out_spikes, l1_spike_sum, actual_time_steps, hw_metrics
