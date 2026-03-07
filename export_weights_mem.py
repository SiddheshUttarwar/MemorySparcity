"""
export_weights_mem.py
=====================
Software-Hardware Bridge: Export trained PyTorch CSNN weights into Verilog-compatible
`.mem` hex files for loading into quantized_sram.v via $readmemh.

Usage:
    python export_weights_mem.py                              # Uses default best_sparse_model.pth
    python export_weights_mem.py --model my_model.pth         # Custom model path
    python export_weights_mem.py --outdir mem_weights          # Custom output directory

Output:
    Creates one .mem file per layer in the output directory:
        conv1_weights.mem   (32 filters × 50 values = 1600 entries)
        conv2_weights.mem   (64 filters × 800 values = 51200 entries)
        fc1_weights.mem     (128 × 3136 = 401408 entries)
        fc2_weights.mem     (10 × 128 = 1280 entries)

    Each line in a .mem file is a 2-character hex value representing a signed INT8 weight
    in two's-complement format, ready for Verilog $readmemh.
"""

import os
import sys
import argparse
import torch
import numpy as np

# Ensure project modules (SRAM, sparse_snn_model, etc.) are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def quantize_to_int8(weight_tensor):
    """
    Quantize a floating-point weight tensor to signed INT8 [-128, 127].
    Uses symmetric quantization: scale = max(|w|) / 127.
    Returns: numpy array of int8 values, scale factor.
    """
    w = weight_tensor.detach().cpu().float()
    w_max = w.abs().max()
    if w_max < 1e-6:
        return np.zeros(w.numel(), dtype=np.int8), 1.0
    
    scale = w_max.item() / 127.0
    q = torch.round(w / scale).clamp(-128, 127).to(torch.int8)
    return q.numpy().flatten(), scale


def int8_to_hex(val):
    """
    Convert a signed INT8 value to a 2-character hex string (two's complement).
    Examples: 0 -> '00', 127 -> '7f', -1 -> 'ff', -128 -> '80'
    """
    return format(int(val) & 0xFF, '02x')


def export_layer_to_mem(weight_tensor, filepath, layer_name):
    """
    Quantize a weight tensor to INT8 and write it as a Verilog .mem file.
    Each line = one hex byte (two's complement signed INT8).
    """
    q_weights, scale = quantize_to_int8(weight_tensor)
    
    with open(filepath, 'w') as f:
        f.write(f"// {layer_name} weights - INT8 quantized (scale={scale:.6f})\n")
        f.write(f"// Shape: {list(weight_tensor.shape)}, Total entries: {len(q_weights)}\n")
        f.write(f"// Load in Verilog: $readmemh(\"{os.path.basename(filepath)}\", mem);\n")
        f.write(f"// Dequantize: float_weight = int8_value * {scale:.6f}\n\n")
        
        for val in q_weights:
            f.write(int8_to_hex(val) + '\n')
    
    # Summary statistics
    nonzero = np.count_nonzero(q_weights)
    sparsity = 1.0 - (nonzero / len(q_weights))
    print(f"  {layer_name:20s} | shape {str(list(weight_tensor.shape)):25s} | "
          f"{len(q_weights):>8d} entries | scale {scale:.6f} | "
          f"sparsity {sparsity*100:.1f}% | -> {os.path.basename(filepath)}")
    
    return scale


def main():
    parser = argparse.ArgumentParser(description="Export PyTorch SNN weights to Verilog .mem files")
    parser.add_argument('--model', type=str, default='best_sparse_model.pth',
                        help='Path to trained model checkpoint (.pth)')
    parser.add_argument('--outdir', type=str, default='mem_weights',
                        help='Output directory for .mem files')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.outdir, exist_ok=True)

    # Load model checkpoint
    print(f"\n{'='*80}")
    print(f"  Software-Hardware Bridge: PyTorch Weight -> Verilog .mem Export")
    print(f"{'='*80}")
    print(f"  Model:  {args.model}")
    print(f"  Output: {args.outdir}/")
    print(f"{'='*80}\n")

    if not os.path.exists(args.model):
        print(f"ERROR: Model file '{args.model}' not found!")
        print("  Train the sparse model first: python train_sparse.py")
        return

    # Import model classes so torch.load can unpickle full model objects
    try:
        from snn_model import *       # noqa: F403
    except Exception:
        pass
    try:
        from sparse_snn_model import *  # noqa: F403
    except Exception:
        pass

    checkpoint = torch.load(args.model, map_location='cpu', weights_only=False)
    
    # Handle multiple checkpoint formats:
    # 1. Raw state_dict (OrderedDict of tensors)
    # 2. Dict with 'model_state_dict' or 'state_dict' key
    # 3. Full model object (has .state_dict() method)
    if hasattr(checkpoint, 'state_dict'):
        # Full model object was saved with torch.save(model, ...)
        state_dict = checkpoint.state_dict()
    elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Define the layers to export (matches LeNet5_Sparse_CSNN architecture)
    layers = [
        ('conv1.weight', 'conv1_weights'),
        ('conv2.weight', 'conv2_weights'),
        ('fc1.weight',   'fc1_weights'),
        ('fc2.weight',   'fc2_weights'),
    ]

    print("  Exporting layers:\n")
    scales = {}
    
    for key, name in layers:
        if key not in state_dict:
            print(f"  WARNING: '{key}' not found in checkpoint, skipping.")
            continue
        
        weight = state_dict[key]
        filepath = os.path.join(args.outdir, f"{name}.mem")
        scale = export_layer_to_mem(weight, filepath, name)
        scales[name] = scale

    # Write a scale manifest file for dequantization reference
    manifest_path = os.path.join(args.outdir, 'quantization_scales.txt')
    with open(manifest_path, 'w') as f:
        f.write("# Quantization Scale Manifest\n")
        f.write("# To dequantize: float_value = int8_value * scale\n\n")
        for name, scale in scales.items():
            f.write(f"{name}: {scale:.8f}\n")

    # Write a Verilog include snippet showing how to load the weights
    verilog_snippet_path = os.path.join(args.outdir, 'load_weights.vh')
    with open(verilog_snippet_path, 'w') as f:
        f.write("// Auto-generated Verilog weight loading snippet\n")
        f.write("// Include this in your testbench or top-level initial block\n\n")
        f.write("initial begin\n")
        for key, name in layers:
            if name in scales:
                f.write(f'    $readmemh("{name}.mem", {name.replace("_weights", "")}_sram.mem);\n')
        f.write("end\n")

    total_params = sum(state_dict[k].numel() for k, _ in layers if k in state_dict)
    total_bytes = total_params  # 1 byte per INT8 weight
    
    print(f"\n{'='*80}")
    print(f"  Export complete!")
    print(f"  Total parameters: {total_params:,d}")
    print(f"  Total SRAM size:  {total_bytes:,d} bytes ({total_bytes/1024:.1f} KB)")
    print(f"  Scale manifest:   {manifest_path}")
    print(f"  Verilog snippet:  {verilog_snippet_path}")
    print(f"{'='*80}\n")
    print(f"  To load in Verilog testbench:")
    print(f'    $readmemh("conv1_weights.mem", conv1_sram.mem);')
    print(f'    $readmemh("conv2_weights.mem", conv2_sram.mem);')
    print(f'    $readmemh("fc1_weights.mem",   fc1_sram.mem);')
    print(f'    $readmemh("fc2_weights.mem",   fc2_sram.mem);\n')


if __name__ == '__main__':
    main()
