import os
import torch
import torch.nn as nn
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

    epochs = 5
    lr = 1e-3
    lambda_reg = 1e-5 # L1 Spike Penalty Weight
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    print("\n--- Starting SPARSE CSNN Training (STBP) ---")
    start_time_global = time.time()
    
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
        sram_efficiency = train_acc / (total_network_spikes / total + 1e-9)
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
        val_sram_efficiency = val_acc / (val_network_spikes / test_total + 1e-9)
        val_avg_t = val_time_steps / len(test_loader)
        
        gate_sparsity = 100. * (epoch_hw_kept / (epoch_hw_in + 1e-9))
        
        print(f"\nEpoch [{epoch+1}/{epochs}]")
        print(f"  Train -> Loss: {train_loss:.4f} | Acc: {train_acc:.2f}% | T-Avg: {avg_t:.1f} steps | SRAM Eff: {sram_efficiency:.4f}")
        print(f"  Test  -> Loss: {val_loss:.4f} | Acc: {val_acc:.2f}% | T-Avg: {val_avg_t:.1f} steps | SRAM Eff: {val_sram_efficiency:.4f}")
        print(f"  Spikes-> Train Total: {total_network_spikes:,.0f} | Val Total: {val_network_spikes:,.0f}")
        print(f"  HW Sim-> Input Gate Kept: {gate_sparsity:.1f}% ({epoch_hw_kept:,.0f}/{epoch_hw_in:,.0f}) | CS_asserts: {epoch_hw_cs:,.0f} | MAC_ops: {epoch_hw_mac:,.0f}")

    print(f"\nSparse Training Completed in {time.time() - start_time_global:.2f} seconds!")

if __name__ == "__main__":
    train_sparse()
