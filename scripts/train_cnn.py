"""
CNN Training Script for Devanagari Handwritten Character Recognition
Uses PyTorch with a CNN architecture that learns spatial features (edges, curves, loops)
instead of raw pixel matching like Random Forest.

Architecture:
  Input: 32x32 grayscale image
  Conv1(32 filters, 3x3) -> BatchNorm -> ReLU -> MaxPool(2x2)
  Conv2(64 filters, 3x3) -> BatchNorm -> ReLU -> MaxPool(2x2)
  Conv3(128 filters, 3x3) -> BatchNorm -> ReLU
  Flatten -> Dropout(0.5) -> FC(256) -> ReLU -> FC(46)
  
This architecture understands shapes — not just pixel locations.
"""

import os
import json
import random
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ─── Model Definition ────────────────────────────────────────────────────────

class DevanagariCNN(nn.Module):
    def __init__(self, num_classes=46):
        super(DevanagariCNN, self).__init__()
        
        # Convolutional feature extractor
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),   # 32x32 -> 16x16
            nn.Dropout2d(0.25),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),   # 16x16 -> 8x8
            nn.Dropout2d(0.25),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ─── Dataset ─────────────────────────────────────────────────────────────────

class DevanagariDataset(Dataset):
    def __init__(self, root_dir, class_names, transform=None, max_per_class=None):
        self.transform = transform
        self.samples = []
        self.labels = []
        self.class_to_idx = {c: i for i, c in enumerate(class_names)}
        
        for class_name in class_names:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            files = [f for f in os.listdir(class_dir) if f.lower().endswith('.png')]
            if max_per_class:
                random.shuffle(files)
                files = files[:max_per_class]
            for fname in files:
                self.samples.append(os.path.join(class_dir, fname))
                self.labels.append(self.class_to_idx[class_name])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert('L')
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

# ─── Training ─────────────────────────────────────────────────────────────────

def train():
    random.seed(42)
    torch.manual_seed(42)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(project_dir, "data", "DevanagariHandwrittenCharacterDataset")
    train_dir = os.path.join(data_dir, "Train")
    test_dir  = os.path.join(data_dir, "Test")
    
    if not os.path.exists(train_dir):
        print("Dataset not found. Please run download_data.py first.")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get class list from train directory
    class_names = sorted([d for d in os.listdir(train_dir)
                          if os.path.isdir(os.path.join(train_dir, d))])
    num_classes = len(class_names)
    print(f"Found {num_classes} classes")
    
    # Save class names for app.py
    class_map_path = os.path.join(project_dir, "models", "cnn_class_names.json")
    with open(class_map_path, 'w') as f:
        json.dump(class_names, f)
    print(f"Class names saved to {class_map_path}")
    
    # Augmentation transforms for training
    train_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.85, 1.15)),
        transforms.ToTensor(),          # Converts to [0,1] float tensor shape (1,32,32)
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    print("\nLoading training dataset...")
    train_dataset = DevanagariDataset(train_dir, class_names, transform=train_transform)
    test_dataset  = DevanagariDataset(test_dir,  class_names, transform=test_transform)
    
    print(f"  Train samples: {len(train_dataset):,}")
    print(f"  Test samples:  {len(test_dataset):,}")
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_dataset,  batch_size=256, shuffle=False, num_workers=0)
    
    # Build model
    model = DevanagariCNN(num_classes=num_classes).to(device)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    
    # Train
    num_epochs = 10
    best_acc = 0.0
    model_path = os.path.join(project_dir, "models", "devanagari_cnn_model.pth")
    
    print(f"\n{'='*55}")
    print(f"Training CNN for {num_epochs} epochs...")
    print(f"{'='*55}\n")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if (batch_idx + 1) % 50 == 0:
                print(f"  Epoch [{epoch+1}/{num_epochs}] Step [{batch_idx+1}/{len(train_loader)}]"
                      f"  Loss: {running_loss/(batch_idx+1):.4f}"
                      f"  Train Acc: {100.*correct/total:.1f}%")
        
        train_acc = 100. * correct / total
        
        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        print(f"\nEpoch [{epoch+1}/{num_epochs}]  "
              f"Train Acc: {train_acc:.2f}%  |  "
              f"Val Acc: {val_acc:.2f}%  |  "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'class_names': class_names
            }, model_path)
            print(f"  [SAVED] New best model! Val Acc = {val_acc:.2f}%\n")
        
        scheduler.step()
    
    print(f"\n{'='*55}")
    print(f"Training complete! Best validation accuracy: {best_acc:.2f}%")
    print(f"Model saved to: {model_path}")
    print(f"{'='*55}")

if __name__ == "__main__":
    train()
