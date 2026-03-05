import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time

class NMNISTDatasetBaseline(Dataset):
    def __init__(self, data_dir, split='train'):
        pattern = os.path.join(data_dir, f'{split}_*.npz')
        self.files = glob.glob(pattern)
        print(f"Loaded {len(self.files)} samples for {split} split.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        spike_tensor = torch.from_numpy(data['data']).float() # [T, C, H, W]
        
        # KEY DIFFERENCE: Collapse the Time Dimension to make it a static 2D Image!
        # By taking the sum over time (T=20), we convert the SNN data into a standard ANN image.
        # Shape becomes [C=2, H=28, W=28]
        static_frame = spike_tensor.sum(dim=0)
        
        label = int(data['digit'])
        return static_frame, label

class FastCNN(nn.Module):
    """ Standard Non-Spiking Convolutional Neural Network """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 14x14
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training Fast Standard CNN Baseline on {device}...")
    
    train_dataset = NMNISTDatasetBaseline('preprocessed_data_native', split='train')
    test_dataset = NMNISTDatasetBaseline('preprocessed_data_native', split='test')
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    if len(train_dataset) == 0:
        print("No training data found. Make sure preprocess_dataset.py completed successfully.")
        return

    model = FastCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    epochs = 3
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        correct, total = 0, 0
        total_loss = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        train_acc = 100. * correct / total
        
        # Validation
        model.eval()
        test_correct, test_total = 0, 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()
                
        test_acc = 100. * test_correct / test_total
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {total_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")
        
    print(f"Fast Baseline Training Completed in {time.time() - start_time:.2f} seconds!")

if __name__ == "__main__":
    main()
