import torch
import torch.nn as nn
import math
from SRAM import SRAMWeightMemory

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
    def __init__(self, in_channels=2, num_classes=10):
        super().__init__()
        self.beta = 0.9
        self.v_th = 1.0 # Restored to 1.0 to prevent dying neurons
        self.rho = 0.05  # Reduced threshold scale to prevent total suppression

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=5, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.lif1 = LIFNodeSTBP_Sparse(beta=self.beta, v_threshold=self.v_th, rho=self.rho)
        self.pool1 = nn.AvgPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.lif2 = LIFNodeSTBP_Sparse(beta=self.beta, v_threshold=self.v_th, rho=self.rho)
        self.pool2 = nn.AvgPool2d(2, 2)

        self.flatten = nn.Flatten()
        
        # Dropout to combat the specific Overfitting observed during training
        self.dropout = nn.Dropout(0.5) 
        
        self.fc1 = nn.Linear(64 * 7 * 7, 128, bias=False)
        self.lif3 = LIFNodeSTBP_Sparse(beta=self.beta, v_threshold=self.v_th, rho=self.rho)
        
        self.fc2 = nn.Linear(128, num_classes, bias=False)
        self.lif4 = LIFNodeSTBP_Sparse(beta=self.beta, v_threshold=self.v_th, rho=self.rho)

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

    def forward(self, x_seq, early_exit=True, confidence_margin=0.9):
        B, T, C, H, W = x_seq.shape
        device = x_seq.device
        
        # Initialize membrane potentials and dynamic thresholds
        v1 = torch.zeros(B, 32, H, W, device=device)
        v_th1 = torch.full((B, 32, H, W), self.v_th, device=device)
        
        v2 = torch.zeros(B, 64, H//2, W//2, device=device)
        v_th2 = torch.full((B, 64, H//2, W//2), self.v_th, device=device)
        
        v3 = torch.zeros(B, 128, device=device)
        v_th3 = torch.full((B, 128), self.v_th, device=device)
        
        v4 = torch.zeros(B, 10, device=device)
        v_th4 = torch.full((B, 10), self.v_th, device=device)
        
        # --- HYBRID SNN GATEKEEPER STATE (Input Interface) ---
        imp_cnt = torch.zeros(B, C, H, W, device=device)
        win_tick = 5
        imp_thresh = 1.0 # Requires at least 1 previous spike to be considered 'important'
        
        last_spiked = torch.zeros(B, C, H, W, dtype=torch.bool, device=device)
        repeat_count = torch.zeros(B, C, H, W, device=device)
        max_repeats = 1
        
        # Hardware Counters for Training Logging
        hw_cs_asserts = 0
        hw_mac_ops = 0
        hw_total_raw_in = 0
        hw_kept_in = 0
        
        out_spikes_seq = []
        l1_spike_sum = torch.tensor(0.0, device=device, requires_grad=True)
        
        # Array to track cumulative SRAM Memory Fetches over time
        cumulative_sram_reads = []
        current_cumulative_reads = 0
        
        actual_time_steps = T
        
        # Step-wise unrolling to allow Temporal Early-Exit
        for t in range(T):
            x_raw_t = x_seq[:, t, ...]
            hw_total_raw_in += (x_raw_t > 0).sum().item()
            
            # 1. Importance Monitor (Update Counters)
            imp_cnt = imp_cnt + x_raw_t
            if t > 0 and t % win_tick == 0:
                imp_cnt = torch.floor(imp_cnt / 2.0) # Bit-shift decay
            
            imp_keep = (imp_cnt >= imp_thresh)
            
            # 2. Burst Redundancy Monitor
            is_spike = (x_raw_t > 0)
            repeat_count = torch.where(is_spike & last_spiked, repeat_count + 1, torch.zeros_like(repeat_count))
            last_spiked = is_spike
            corr_keep = (repeat_count <= max_repeats)
            
            # 3. Sparsity Controller (Gatekeeper Decision)
            gate_keep = is_spike & imp_keep & corr_keep
            x_t = gate_keep.float() * x_raw_t
            
            # Increment Hardware Activity Proxies based on exactly what breached the gate
            active_spikes = gate_keep.sum().item()
            hw_kept_in += active_spikes
            hw_cs_asserts += active_spikes # Memory Asserted only on Kept Spike
            hw_mac_ops += active_spikes * (5*5 * 32) # Approx MAC ops triggered per kept spike against Conv1 weights
            
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
            
            # Apply dropout during training to randomly kill 50% of signals!
            x_t = self.dropout(x_t)
            
            x_t = self.fc1(x_t)
            spike3, v3, v_th3 = self.lif3(x_t, v3, v_th3)
            
            x_t = self.fc2(spike3)
            out_spike, v4, v_th4 = self.lif4(x_t, v4, v_th4)
            
            out_spikes_seq.append(out_spike)
            # Calculate total spikes (and resulting internal SRAM fetches) for this exact timestep
            step_internal_spikes = spike1.sum() + spike2.sum() + spike3.sum() + out_spike.sum()
            l1_spike_sum = l1_spike_sum + step_internal_spikes
            
            # Step-wise Cumulative Trace
            current_cumulative_reads += active_spikes + int(step_internal_spikes.item())
            cumulative_sram_reads.append(current_cumulative_reads)
            
            if early_exit and t >= 3:
                # Calculate mean rate so far
                current_spikes = torch.stack(out_spikes_seq, dim=1) 
                current_rate = current_spikes.mean(dim=1)
                probs = torch.softmax(current_rate * 5.0, dim=1) # Temperature scaling
                max_probs, _ = probs.max(dim=1)
                
                # If ALL items in batch are confident, EXIT EARLY to save SRAM reads!
                if (max_probs > confidence_margin).all():
                    actual_time_steps = t + 1
                    break
                    
        # Pad sequence with zeros if early exit happened to maintain tensor shapes downstream
        while len(out_spikes_seq) < T:
            out_spikes_seq.append(torch.zeros_like(out_spikes_seq[-1]))
            
        out_spikes = torch.stack(out_spikes_seq, dim=1)
        # Calculate spike rate over the ACTUAL simulated steps, not T
        spike_rate = out_spikes[:, :actual_time_steps, :].mean(dim=1)
        
        # Return Hardware Counter Dictionary
        hw_metrics = {
            'cs_asserts': hw_cs_asserts,
            'mac_ops': hw_mac_ops,
            'total_in': hw_total_raw_in,
            'kept_in': hw_kept_in,
            'cumulative_reads_over_time': cumulative_sram_reads
        }
        
        return spike_rate, out_spikes, l1_spike_sum, actual_time_steps, hw_metrics
