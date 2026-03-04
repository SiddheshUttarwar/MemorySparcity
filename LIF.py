import numpy as np


class LIF:
    """
    Single-neuron LIF cell.

    Default parameters are normalized for float inputs.
    """

    def __init__(self, threshold=1.0, beta=0.9, leak=0.0, v_rest=0.0, v_reset=0.0, dtype=np.float32):
        self.threshold = float(threshold)
        self.beta = float(beta)
        self.leak = float(leak)
        self.v_rest = float(v_rest)
        self.v_reset = float(v_reset)
        self.dtype = dtype
        self.v = self.dtype(self.v_rest)

    def reset(self):
        self.v = self.dtype(self.v_rest)

    def step(self, I_synaptic):
        """
        I_synaptic: scalar synaptic current.
        """
        # Leaky integration around resting potential.
        self.v = self.beta * (self.v - self.v_rest) + self.v_rest + self.dtype(I_synaptic) - self.leak

        if self.v >= self.threshold:
            self.v = self.dtype(self.v_reset)
            return 1
        return 0


class LIFLayer:
    """
    Vectorized LIF layer for batch simulation.
    """

    def __init__(self, size, threshold=1.0, beta=0.9, leak=0.0, v_rest=0.0, v_reset=0.0, dtype=np.float32):
        self.size = int(size)
        self.threshold = float(threshold)
        self.beta = float(beta)
        self.leak = float(leak)
        self.v_rest = float(v_rest)
        self.v_reset = float(v_reset)
        self.dtype = dtype
        self.v = None

    def reset(self, batch_size):
        self.v = np.full((batch_size, self.size), self.v_rest, dtype=self.dtype)

    def step(self, current):
        if self.v is None or self.v.shape[0] != current.shape[0]:
            self.reset(current.shape[0])

        self.v = self.beta * (self.v - self.v_rest) + self.v_rest + current.astype(self.dtype) - self.leak
        spikes = (self.v >= self.threshold).astype(self.dtype)
        self.v = np.where(spikes > 0, self.v_reset, self.v)
        return spikes
