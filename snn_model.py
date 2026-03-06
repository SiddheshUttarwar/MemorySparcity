import torch
import torch.nn as nn
import math
from SRAM import SRAMWeightMemory

class SurrogateFastSigmoid(torch.autograd.Function):
    """
    Surrogate Gradient for the non-differentiable Heaviside step function.
    Using Fast Sigmoid H(x) = x / (1 + |x|) surrogate gradient function as requested.
    """
    alpha = 10.0  # Steepness of the surrogate gradient. 10.0 is recommended for deep SNNs.

    @staticmethod
    def forward(ctx, input_tensor, threshold):
        ctx.save_for_backward(input_tensor, threshold)
        # Forward pass: True binary spike if V >= V_th
        out = torch.zeros_like(input_tensor)
        out[input_tensor >= threshold] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, threshold = ctx.saved_tensors
        # Backward pass: "Fake" gradient using Fast Sigmoid derivative
        # dS/dU = 1 / (1 + alpha * |U - V_th|)^2
        grad_input = grad_output.clone()
        surrogate_grad = 1.0 / (1.0 + SurrogateFastSigmoid.alpha * torch.abs(input_tensor - threshold)).pow(2)
        return grad_input * surrogate_grad, None

class LIFNodeSTBP(nn.Module):
    """
    Leaky Integrate-and-Fire (LIF) Neuron Node using Surrogate Gradient Backpropagation (STBP).
    Handles temporal dynamics natively across a sequence of time steps.
    """
    def __init__(self, beta=0.9, v_threshold=1.0, v_rest=0.0):
        super().__init__()
        # Beta defines membrane decay directly: V(t) = V(t-1) * beta + X(t)
        self.beta = beta
        self.v_threshold = torch.tensor(v_threshold)
        self.v_rest = v_rest

    def forward(self, x_seq):
        """
        x_seq: [Batch, Time, Channels, Height, Width] or [Batch, Time, Features]
        Returns: spike_seq [Batch, Time, ...]
        """
        device = x_seq.device
        batch_size = x_seq.size(0)
        time_steps = x_seq.size(1)
        spatial_dims = x_seq.shape[2:]

        # Initialize Membrane Potential
        v = torch.full((batch_size, *spatial_dims), self.v_rest, device=device, dtype=torch.float32)
        
        spike_seq = []

        for t in range(time_steps):
            x_t = x_seq[:, t, ...]
            
            # 1. Integrate input current
            v = v * self.beta + x_t
            
            # 2. Generate Spike using Fast Sigmoid Surrogate Gradient
            spike = SurrogateFastSigmoid.apply(v, self.v_threshold.to(device))
            
            # 3. Reset Membrane Potential (Soft reset preferred for STBP accuracy)
            v = v - spike * self.v_threshold.to(device)
            
            spike_seq.append(spike)
            
        return torch.stack(spike_seq, dim=1)

class LeNet5_CSNN(nn.Module):
    """
    LeNet-5 style Convolutional Spiking Neural Network using STBP.
    Treats the network inherently as a recurrent graph over the time dimension.
    """
    def __init__(self, in_channels=2, num_classes=10):
        """
        in_channels: 2 for natively event based N-MNIST (ON/OFF polarities)
        """
        super().__init__()
        
        # Hyperparameters chosen for stable deep STBP
        self.beta = 0.9
        self.v_th = 1.0

        # Conv Layer 1 (5x5 kernel as recommended for LeNet-5)
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=5, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.lif1 = LIFNodeSTBP(beta=self.beta, v_threshold=self.v_th)
        self.pool1 = nn.AvgPool2d(2, 2) # Average pooling is often more stable in SNNs but MaxPool works too (28x28 -> 14x14)

        # Conv Layer 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.lif2 = LIFNodeSTBP(beta=self.beta, v_threshold=self.v_th)
        self.pool2 = nn.AvgPool2d(2, 2) # 14x14 -> 7x7

        # Fully Connected Layer (128 hidden units as recommended)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 7 * 7, 128, bias=False)
        self.lif3 = LIFNodeSTBP(beta=self.beta, v_threshold=self.v_th)
        
        # Output Readout Layer
        self.fc2 = nn.Linear(128, num_classes, bias=False)
        self.lif4 = LIFNodeSTBP(beta=self.beta, v_threshold=self.v_th)

        # Kaiming He Initialization
        self._initialize_weights()

        # Map newly initialized weights to the localized SRAM arrays
        self._init_sram()

    def _init_sram(self):
        """Initializes physical SRAM structures mapping parametric neuron layouts."""
        self.sram_memories = {}
        # Conv1: 32 out, 2 in, 5x5 kernels -> [32 rows, 50 cols]
        self.sram_memories['conv1'] = SRAMWeightMemory(rows=32, cols=2*5*5)
        # Conv2: 64 out, 32 in, 5x5 kernels -> [64 rows, 800 cols]
        self.sram_memories['conv2'] = SRAMWeightMemory(rows=64, cols=32*5*5)
        # FC1: 128 out, 64*7*7 in -> [128 rows, 3136 cols]
        self.sram_memories['fc1'] = SRAMWeightMemory(rows=128, cols=64*7*7)
        # Output: 10 out, 128 in -> [10 rows, 128 cols]
        self.sram_memories['fc2'] = SRAMWeightMemory(rows=10, cols=128)
        
        # Hydrate SRAM immediately
        self.sync_to_sram()

    def sync_to_sram(self):
        """Writes current Pytorch weights into the static SRAM blocks."""
        # Conv weights are detached, pushed to CPU, and flattened to standard 2D arrays
        c1_w = self.conv1.weight.detach().cpu().numpy().reshape(32, -1)
        self.sram_memories['conv1'].load_from_array(c1_w)
        
        c2_w = self.conv2.weight.detach().cpu().numpy().reshape(64, -1)
        self.sram_memories['conv2'].load_from_array(c2_w)
        
        fc1_w = self.fc1.weight.detach().cpu().numpy()
        self.sram_memories['fc1'].load_from_array(fc1_w)
        
        fc2_w = self.fc2.weight.detach().cpu().numpy()
        self.sram_memories['fc2'].load_from_array(fc2_w)

    def sync_from_sram(self):
        """Fetches weights from SRAM and writes them forcefully back onto the Pytorch execution graph."""
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

    def forward(self, x):
        """
        x shape expected: [Batch, Time, Channels, Height, Width]
        By convention for deep SNNs, we unfold Time, apply Conv2d to batch*time, then reshape for LIF.
        """
        B, T, C, H, W = x.shape
        
        # Collapse Batch and Time for Convolution operations
        x = x.view(B * T, C, H, W)
        
        # Layer 1
        x = self.conv1(x)
        x = self.bn1(x)
        _, C1, H1, W1 = x.shape
        x = x.view(B, T, C1, H1, W1)
        x = self.lif1(x)
        x = x.view(B * T, C1, H1, W1)
        x = self.pool1(x)

        # Layer 2
        x = self.conv2(x)
        x = self.bn2(x)
        _, C2, H2, W2 = x.shape
        x = x.view(B, T, C2, H2, W2)
        x = self.lif2(x)
        x = x.view(B * T, C2, H2, W2)
        x = self.pool2(x)
        
        # Flatten
        x = self.flatten(x)
        _, F1 = x.shape
        x = x.view(B, T, F1)
        
        # FC Layer 1
        x = x.view(B * T, F1)
        x = self.fc1(x)
        _, F2 = x.shape
        x = x.view(B, T, F2)
        x = self.lif3(x)
        
        # FC Output Layer
        x = x.view(B * T, F2)
        x = self.fc2(x)
        x = x.view(B, T, -1)
        out_spikes = self.lif4(x)
        
        # CrossEntropy on Spike Target (Firing Frequency over Time)
        spike_rate = out_spikes.mean(dim=1) 
        return spike_rate, out_spikes

if __name__ == "__main__":
    # Test execution
    print("--- LeNet-5 Convolutional SNN via STBP ---")
    model = LeNet5_CSNN(in_channels=2, num_classes=10)
    
    # Fake input tensor matching optimal TimeSteps (T=10) requested:
    # [Batch, TimeBins, Channels, H, W] -> e.g. [2, 10, 2, 28, 28]
    fake_input_tensor = torch.rand(2, 10, 2, 28, 28)
    fake_input_tensor.requires_grad = True
    
    # Forward Pass
    spike_rate, output_spikes = model(fake_input_tensor)
    
    print(f"Input Shape: {fake_input_tensor.shape}")
    print(f"Output Spike Rate Shape (Readout): {spike_rate.shape}")
    print(f"Total Output Spikes produced in Sequence: {output_spikes.sum().item()}")
    
    # Fake Loss Calculation using Spike-based MSE
    target_spikes = torch.zeros_like(spike_rate)
    target_spikes[0, 5] = 1.0 # Target class 5 for item 0
    target_spikes[1, 2] = 1.0 # Target class 2 for item 1
    
    loss_fn = nn.MSELoss()
    loss = loss_fn(spike_rate, target_spikes)
    loss.backward()
    
    print("\n--- STBP Gradient Check ---")
    print(f"Loss: {loss.item():.4f}")
    print(f"Conv1 weight grad norm: {model.conv1.weight.grad.norm().item():.4f}")
    print("LeNet-5 SNN initialized properly with Fast Sigmoid Surrogate Gradients & Kaiming initialization!")
