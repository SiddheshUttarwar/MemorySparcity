import numpy as np

class GatedSNN:
    def __init__(self, num_inputs, num_neurons, threshold=256, leak=4):
        self.num_inputs = num_inputs
        self.num_neurons = num_neurons
        
        # 1. Synaptic Weight SRAM (8-bit signed integers)
        self.weights = np.random.randint(-20, 50, size=(num_inputs, num_neurons), dtype=np.int8)
        
        # 2. Neuron Array (Hardware-Ready LIF)
        self.v_mem = np.zeros(num_neurons, dtype=np.int16)
        self.threshold = threshold
        self.leak = leak
        
        # 3. Sparsity Gating Mask (1 = Active, 0 = Gated/Pruned)
        self.gating_mask = np.ones(num_inputs, dtype=np.uint8)

    def process_time_step(self, input_spikes):
        """
        input_spikes: Binary vector from the 'Spike Router'
        """
        # --- PHASE 1: Sparsity Gating (The Hardware 'Valve') ---
        # Only process spikes where the gating mask is 1
        active_spikes = input_spikes & self.gating_mask
        active_indices = np.where(active_spikes > 0)[0]
        
        # This 'active_indices' is what your Hardware Router would 
        # send to the SRAM Address Bus.
        
        # --- PHASE 2: Synaptic Integration (SRAM Fetch) ---
        current_input = np.zeros(self.num_neurons, dtype=np.int16)
        if len(active_indices) > 0:
            # Sum up weights only for non-gated, spiking inputs
            current_input = np.sum(self.weights[active_indices], axis=0, dtype=np.int16)

        # --- PHASE 3: LIF Engine (Membrane Update) ---
        # Add Input
        self.v_mem += current_input
        
        # Linear Leak (Silicon-efficient)
        self.v_mem = np.maximum(self.v_mem - self.leak, 0)
        
        # Fire & Reset
        output_spikes = (self.v_mem >= self.threshold).astype(np.uint8)
        self.v_mem[output_spikes > 0] = 0 # Hard Reset
        
        return output_spikes

# --- Simulation Run ---
snn = GatedSNN(num_inputs=10, num_neurons=4)

# Manually 'Prune' index 5 to test the Gating Logic
snn.gating_mask[5] = 0 

# Input spike train (Spikes at index 0, 5, and 9)
input_data = np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 1], dtype=np.uint8)

# Run Step
output = snn.process_time_step(input_data)

print(f"Gating Mask: {snn.gating_mask}")
print(f"Output Spikes: {output}")
print(f"Neuron Potentials: {snn.v_mem}")