import os
import torch
import torch.nn as nn
try:
    from torchinfo import summary
except ImportError:
    summary = None
from torch.utils.data import DataLoader
from train import NMNISTDataset
from sparse_snn_model import LeNet5_Sparse_CSNN
import time

def train_sparse():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing SPARSE STBP Training on Device: {DEVICE}")

    train_dataset = NMNISTDataset(data_dir='preprocessed_data_native', split='train')
    test_dataset = NMNISTDataset(data_dir='preprocessed_data_native', split='test')
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    if len(train_dataset) == 0:
        print("No training data found. Make sure preprocess_dataset.py completed successfully.")
        return

    # Initialize Sparse Model
    model = LeNet5_Sparse_CSNN(in_channels=2, num_classes=10).to(DEVICE)
    
    # Optional Visualization
    if summary is not None:
        print("\n--- Model Architecture Visualization ---")
        # Dummy input for torchinfo (Batch, Time, Channels, Height, Width)
        summary(model, input_size=(1, 20, 2, 28, 28), device=DEVICE)
        print("--------------------------------------\n")

    epochs = 10 # Increased to 10 for better convergence
    lr = 2e-3   # Slightly higher initial LR
    lambda_reg = 1e-7 # Reduced from 1e-5 to prevent Total Spike Death
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Cosine Annealing Learning Rate Scheduler for late-stage accuracy squeezing
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    criterion = nn.CrossEntropyLoss()

    print("\n--- Starting SPARSE CSNN Training (STBP) ---")
    start_time_global = time.time()
    
    # Global Best model tracking across all epochs
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        total_network_spikes = 0.0
        total_time_steps = 0
        epoch_hw_cs = 0
        epoch_hw_mac = 0
        epoch_hw_in = 0
        epoch_hw_kept = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            optimizer.zero_grad()
            model.sync_from_sram()
            
            # Forward pass with Early-Exit enabled
            spike_rate, output_spikes, l1_spike_sum, actual_steps, hw_metrics = model(inputs, early_exit=True)
            
            loss_task = criterion(spike_rate, targets)
            # Apply L1 Spike Regularization directly into the gradient graph
            loss_reg = lambda_reg * l1_spike_sum
            loss = loss_task + loss_reg
            
            loss.backward()
            optimizer.step()
            model.sync_to_sram()
            
            total_loss += loss.item()
            _, predicted = spike_rate.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            total_network_spikes += l1_spike_sum.item()
            total_time_steps += actual_steps
            
            epoch_hw_cs += hw_metrics['cs_asserts']
            epoch_hw_mac += hw_metrics['mac_ops']
            epoch_hw_in += hw_metrics['total_in']
            epoch_hw_kept += hw_metrics['kept_in']
            
        train_acc = 100. * correct / total
        train_loss = total_loss / len(train_loader)
        
        # Calculate how many spikes were fired PER INFERENCE on average
        spikes_per_inference_train = total_network_spikes / total
        # SRAM efficiency: accuracy points per 1000 spikes (scaled for readability)
        sram_efficiency = train_acc / (spikes_per_inference_train / 1000.0 + 1e-9)
        avg_t = total_time_steps / len(train_loader)
        
        # Validation
        model.eval()
        model.sync_from_sram()
        test_loss, test_correct, test_total = 0, 0, 0
        val_network_spikes = 0.0
        val_time_steps = 0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                spike_rate, output_spikes, l1_spike_sum, actual_steps, hw_metrics = model(inputs, early_exit=True)
                
                loss = criterion(spike_rate, targets)
                test_loss += loss.item()
                
                _, predicted = spike_rate.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()
                val_network_spikes += l1_spike_sum.item()
                val_time_steps += actual_steps
                
        val_acc = 100. * test_correct / test_total
        val_loss = test_loss / len(test_loader)
        spikes_per_inference_val = val_network_spikes / test_total
        val_sram_efficiency = val_acc / (spikes_per_inference_val / 1000.0 + 1e-9)
        val_avg_t = val_time_steps / len(test_loader)
        
        gate_sparsity = 100. * (epoch_hw_kept / (epoch_hw_in + 1e-9))
        
        # Step the learning rate scheduler
        scheduler.step()
        
        print(f"\nEpoch [{epoch+1}/{epochs}] | LR: {scheduler.get_last_lr()[0]:.6f}")
        print(f"  Train -> Loss: {train_loss:.4f} | Acc: {train_acc:.2f}% | T-Avg: {avg_t:.1f} steps | SRAM Eff: {sram_efficiency:.4f}")
        print(f"  Test  -> Loss: {val_loss:.4f} | Acc: {val_acc:.2f}% | T-Avg: {val_avg_t:.1f} steps | SRAM Eff: {val_sram_efficiency:.4f}")
        print(f"  Spikes-> Train Total: {total_network_spikes:,.0f} | Val Total: {val_network_spikes:,.0f}")
        print(f"  HW Sim-> Input Gate Kept: {gate_sparsity:.1f}% ({epoch_hw_kept:,.0f}/{epoch_hw_in:,.0f}) | CS_asserts: {epoch_hw_cs:,.0f} | MAC_ops: {epoch_hw_mac:,.0f}")

        # Checkpoint the Best Model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"  => [Saving Checkoutpoint]: New Best Accuracy {best_val_acc:.2f}%")
            torch.save(model.state_dict(), "best_sparse_model.pth")

    print(f"\nSparse Training Completed in {time.time() - start_time_global:.2f} seconds!")
    print(f"Best Validation Accuracy achieved: {best_val_acc:.2f}%")
    print(f"Model saved to: best_sparse_model.pth")

if __name__ == "__main__":
    train_sparse()
