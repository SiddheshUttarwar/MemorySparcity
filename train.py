import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from snn_model import LeNet5_CSNN

class NMNISTDataset(Dataset):
    """
    Loads preprocessed pure spatiotemporal spike tensors from disk.
    Expected data structure: [Time, Channels, Height, Width]
    """
    def __init__(self, data_dir, split='train', max_samples=None):
        pattern = os.path.join(data_dir, f'{split}_*.npz')
        self.files = glob.glob(pattern)
        
        # Sort or shuffle if necessary, but we'll let DataLoader shuffle
        if max_samples is not None and len(self.files) > max_samples:
            self.files = self.files[:max_samples]
            
        print(f"Loaded {len(self.files)} samples for {split} split.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Load .npz
        data = np.load(self.files[idx])
        
        # Tensor shape is [T=20, C=2, H=28, W=28], type bool
        spike_tensor = torch.from_numpy(data['data']).float()
        
        # The true digit labels are saved under 'digit'
        label = int(data['digit'])
        
        # Our CSNN is expecting [T, C, H, W] inside a batch [B, T, C, H, W]
        return spike_tensor, label

def train_snn():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing STBP Training on Device: {DEVICE}")

    # 1. Dataset & DataLoaders
    # We use our subset preprocessed_data_native (we saved 500 samples earlier)
    train_dataset = NMNISTDataset(data_dir='preprocessed_data_native', split='train')
    test_dataset = NMNISTDataset(data_dir='preprocessed_data_native', split='test')
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    if len(train_dataset) == 0:
        print("No training data found. Make sure preprocess_dataset.py completed successfully.")
        return

    # 2. Model Initialization
    # The LeNet5_CSNN natively incorporates "Reset by Subtraction" (Soft Reset)
    # in the LIFNodes keeping Beta=0.9 and V_th=1.0 for optimum performance.
    model = LeNet5_CSNN(in_channels=2, num_classes=10).to(DEVICE)

    # 3. Training Hyperparameters
    epochs = 5
    lr = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Cross Enrtopy Loss naturally uses LogSoftmax on the output spike count 
    # ensuring the "highest firing count" strategy is explicitly maximized.
    criterion = nn.CrossEntropyLoss()

    print("\n--- Starting CSNN Training (STBP) ---")
    
    # 4. Training Loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Forward pass: model returns spike_rate (mean spikes over Time), and output_spikes [B, T, Classes]
            spike_rate, output_spikes = model(inputs)
            
            # Rate Coding Target: Predict class with highest spike rate
            loss = criterion(spike_rate, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Calculate accuracy based on highest spike rate
            _, predicted = spike_rate.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        train_acc = 100. * correct / total
        train_loss = total_loss / len(train_loader)
        
        # 5. Validation Loop
        model.eval()
        test_loss = 0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                spike_rate, output_spikes = model(inputs)
                
                loss = criterion(spike_rate, targets)
                test_loss += loss.item()
                
                _, predicted = spike_rate.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()
                
        val_acc = 100. * test_correct / test_total
        val_loss = test_loss / len(test_loader)
        
        print(f"Epoch [{epoch+1}/{epochs}] | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Test Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")

if __name__ == "__main__":
    train_snn()
