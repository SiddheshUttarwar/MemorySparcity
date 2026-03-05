# Neuromorphic Spiking Neural Network (CSNN) on N-MNIST

This repository contains a full pipeline to process the natively event-based N-MNIST dataset and train a deep **Convolutional Spiking Neural Network (CSNN)** using Surrogate Gradient Backpropagation (STBP).

The pipeline involves:
1. `preprocess_dataset.py` - Converts 70,000 raw `.bin` polarity events into static `[T, C, H, W]` Spatio-Temporal matrices.
2. `snn_model.py` - Core PyTorch dynamics implementing a strict LeNet-5 Spiking Architecture over time (`T=20`).
3. `train.py` - Supervised STBP learning loop utilizing Rate Coding Cross Entropy.
4. `train_fast_cnn.py` - A collapsed Artificial Neural Network (ANN) baseline to verify dataset health.

## Quick-Start GPU Training Guide (Windows PowerShell)

If you have an NVIDIA GPU, you can train the full Convolutional SNN rapidly by running the following commands in order.

### 1. Setup the Virtual Environment
Create a clean PyTorch environment utilizing the official GPU CUDA wheels.

```powershell
# 1. Create native isolated environment and bypass Windows Execution protections
python -m venv venv
Set-ExecutionPolicy -ExecutionPolicy UNRESTRICTED -Scope CurrentUser
.\venv\Scripts\Activate.ps1

# 2. Upgrade pip module to avoid local cache errors
python -m pip install --upgrade pip

# 3. Install PyTorch with NVIDIA CUDA 12.1 support (Replace cu121 with your CUDA version if different)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. Install supporting math/visualization libraries
pip install numpy scipy matplotlib
```

### 2. Preprocess the Complete Dataset
N-MNIST consists of 70,000 `.bin` files (`Train.zip` & `Test.zip`). 

Run the multiprocessing builder. This script leverages all your CPU cores to crunch and parse the files into structured `.npz` spike tensors inside a new `preprocessed_data_native/` folder.
*This takes about 1-2 minutes on a fast CPU.*

```powershell
# Ensure your virtual environment is active (.\venv\Scripts\Activate.ps1)
python preprocess_dataset.py
```

### 3. Verify Dataset (Optional Baseline)
Before committing your GPU to the intense SNN temporal simulation loop, you can verify your data extracted correctly by training a standard fast CNN on the static 2D-collapsed version of the frames.

```powershell
python train_fast_cnn.py
```
*(You should see >98% accuracy hit within ~2 minutes)*.

### 4. Train the Spiking Neural Network (CSNN)
Once your data is available and your GPU environment is solid, launch the surrogate gradient trainer. This will iteratively unroll the CSNN $T$-times per sample, applying the Fast Sigmoid surrogate STBP logic via `snntorch` principles native PyTorch operations.

```powershell
python train.py
```

*Note: The script natively searches for `cuda` and will automatically offload tensors to your GPU. Watch your Loss metrics!*
