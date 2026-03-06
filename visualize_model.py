import torch
from sparse_snn_model import LeNet5_Sparse_CSNN
import sys

# Attempt to load visualization libraries as requested
try:
    from torchinfo import summary
    import torchviz
except ImportError:
    print("Please install visualization libraries first in Colab:")
    print("!pip install torchinfo torchviz graphviz")
    sys.exit(1)

def visualize_snn():
    # 1. Instantiate the Model
    # Note: We must explicitly set device to match where the input tensor will live
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading Model on Device: {device}")
    
    model = LeNet5_Sparse_CSNN(in_channels=2, num_classes=10).to(device)
    
    # 2. Print Text-Based Architecture Summary (torchinfo)
    print("\n" + "="*50)
    print("NETWORK ARCHITECTURE (torchinfo)")
    print("="*50)
    # Input tensor shape: (Batch Size, Time Steps, Channels, Height, Width)
    # We use batch=1, T=20, Channels=2 (N-MNIST), H=28, W=28 (after cropping)
    summary(model, input_size=(1, 20, 2, 28, 28), device=device)
    
    # 3. Generate Computation Graph (torchviz)
    print("\n" + "="*50)
    print("COMPUTATION GRAPH (torchviz)")
    print("="*50)
    
    # Generate a dummy input matching the expected dimensions
    dummy_input = torch.randn(1, 20, 2, 28, 28).to(device)
    
    # Run a forward pass to trace the gradient operations
    spike_rate, _, _, _, _ = model(dummy_input, early_exit=False)
    
    # Generate visual graph mapping nodes to their computation equivalents
    graph = torchviz.make_dot(spike_rate, params=dict(list(model.named_parameters())))
    
    # Save the PDF rendering output
    output_filename = "sparse_snn_architecture"
    graph.render(output_filename, format="pdf")
    graph.render(output_filename, format="png")
    
    print(f"Success! Graph natively exported to {output_filename}.pdf and {output_filename}.png")

if __name__ == "__main__":
    visualize_snn()
