# Neuromorphic Spiking Neural Network (CSNN) on N-MNIST

This repository contains a full pipeline to process the natively event-based N-MNIST dataset and train a deep **Convolutional Spiking Neural Network (CSNN)** using Surrogate Gradient Backpropagation (STBP).

## Complete Guide: From Raw Data to GPU Training

Follow these exact steps to set up the environment, process the raw N-MNIST spike events, and train the network on your GPU.

### Step 1: Ensure you have the N-MNIST Dataset
Make sure you have downloaded the N-MNIST dataset. You must have the two zip files sitting perfectly in the root directory of this project:
- `Train.zip`
- `Test.zip`

*(Do not extract them yourself. The Python scripts will read directly from the `.zip` archives to save disk space and file indexing overhead.)*

### Step 2: Set up the PyTorch GPU Environment (Windows PowerShell)
You need an isolated Python environment with NVIDIA CUDA capabilities to execute the Surrogate Gradient training efficiently. Run these exact rules in your PowerShell terminal inside the project directory:

```powershell
# 1. Create a native isolated virtual environment
python -m venv .venv

# 2. Bypass Windows App Execution limits so you can activate the environment
Set-ExecutionPolicy -ExecutionPolicy UNRESTRICTED -Scope CurrentUser

# 3. Activate the virtual environment! (You must do this every time you open a new terminal)
.\.venv\Scripts\Activate.ps1

# 4. Give pip an upgrade
python -m pip install --upgrade pip

# 5. Install PyTorch with NVIDIA CUDA 12.1 support (Change cu121 if your GPU uses a different version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 6. Install the supporting computation libraries
pip install numpy scipy matplotlib
```

### Step 3: Preprocess the 70,000 N-MNIST Events
The N-MNIST dataset format (`x, y, polarity, timestamp`) must be translated into spatial-temporal tensors `[Time, Channels, Height, Width]` for the deep network.

Launch the preprocessing pipeline. It uses multiprocessing across all your CPU cores to crunch 70,000 files in under 2 minutes.

```powershell
# Make sure your environment is active: .\venv\Scripts\Activate.ps1
python preprocess_dataset.py
```
*Wait for this to finish printing "All completely done!". It will create a `preprocessed_data_native` directory.*

### Step 4: Verify the Data Pipeline (Optional Baseline)
Before launching the heavy Spiking Neural Network, you can verify your dataset extracted perfectly by training a standard, non-spiking CNN (which crushes the time dimension perfectly into 2D imagery).

```powershell
python train_fast_cnn.py
```
*(You should see >98% accuracy hit within ~2 minutes. This proves everything works!)*

### Step 5: Train the Convolutional Spiking Neural Network (CSNN)
Now, train the true deep SNN. This script natively locates your NVIDIA GPU, builds the LeNet-5 architecture with `snntorch` style surrogate gradients, and unrolls the physical time dynamics ($T=20$ time steps).

**SRAM Integration Note:** This architecture physically integrates with `SRAM.py`! Before every training iteration, the PyTorch tensors fetch their parameter weights directly from simulated `SRAMWeightMemory` blocks. During backpropagation, the PyTorch gradients are forcefully copied back into the SRAM matrices, ensuring all memory modeling behaves identically to Neuromorphic hardware.

```powershell
python train.py
```

Watch the training epochs and Cross-Entropy Loss print out. Your GPU is now mapping temporal spike behavior linked directly to the SRAM blocks!
