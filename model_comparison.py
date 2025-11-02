"""
CIFAR-10 Model Comparison: MLP vs CNN vs ViT
Compare three different architectures for image classification
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


# Hyperparameters
BATCH_SIZE = 128
LEARNING_RATE = 0.001
NUM_EPOCHS = 20
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# CIFAR-10 classes
CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck')


# ============================================================================
# MODEL DEFINITIONS
# ============================================================================

class MLP(nn.Module):
    """Multi-Layer Perceptron"""
    
    def __init__(self, input_size=3072, hidden_size1=512, hidden_size2=256, num_classes=10):
        super(MLP, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(hidden_size2, num_classes)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


class CNN(nn.Module):
    """Convolutional Neural Network"""
    
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.3)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.4)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class PatchEmbedding(nn.Module):
    """Split image into patches and embed them"""
    
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=256):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e h w -> b (h w) e')
        )
    
    def forward(self, x):
        x = self.projection(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer Encoder Block"""
    
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class ViT(nn.Module):
    """Vision Transformer (Small size for CIFAR-10)"""
    
    def __init__(self, img_size=32, patch_size=4, in_channels=3, num_classes=10,
                 embed_dim=256, depth=6, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
    
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=B)
        x = torch.cat([cls_tokens, x], dim=1)
        
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        cls_token_final = x[:, 0]
        x = self.head(cls_token_final)
        
        return x


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data():
    """Load CIFAR-10 dataset"""
    print("Loading dataset...")
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return train_loader, test_loader


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_one_epoch(model, train_loader, criterion, optimizer, epoch, model_name):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'{model_name} Epoch {epoch+1}/{NUM_EPOCHS}')
    for images, labels in pbar:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{running_loss/total:.4f}',
            'acc': f'{100*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


def evaluate(model, test_loader, criterion):
    """Evaluate model"""
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    test_loss = test_loss / total
    test_acc = 100 * correct / total
    return test_loss, test_acc


def train_model(model, model_name, train_loader, test_loader, num_epochs=NUM_EPOCHS):
    """Complete training loop for a model"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, epoch, model_name)
        test_loss, test_acc = evaluate(model, test_loader, criterion)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"  Test  - Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%")
        print("-" * 60)
    
    training_time = time.time() - start_time
    
    print(f"\n{model_name} Training Complete!")
    print(f"Total training time: {training_time/60:.2f} minutes")
    print(f"Best test accuracy: {max(test_accs):.2f}%")
    
    return {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_losses': test_losses,
        'test_accs': test_accs,
        'training_time': training_time,
        'best_acc': max(test_accs)
    }


def get_class_accuracy(model, test_loader):
    """Calculate per-class accuracy"""
    class_correct = [0] * 10
    class_total = [0] * 10
    
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    class_acc = [100 * class_correct[i] / class_total[i] for i in range(10)]
    return class_acc


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_training_curves(results):
    """Plot training curves for all models"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    epochs = range(1, NUM_EPOCHS + 1)
    
    # Training Loss
    ax = axes[0, 0]
    ax.plot(epochs, results['MLP']['train_losses'], 'b-', label='MLP', linewidth=2)
    ax.plot(epochs, results['CNN']['train_losses'], 'g-', label='CNN', linewidth=2)
    ax.plot(epochs, results['ViT']['train_losses'], 'r-', label='ViT', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Test Loss
    ax = axes[0, 1]
    ax.plot(epochs, results['MLP']['test_losses'], 'b-', label='MLP', linewidth=2)
    ax.plot(epochs, results['CNN']['test_losses'], 'g-', label='CNN', linewidth=2)
    ax.plot(epochs, results['ViT']['test_losses'], 'r-', label='ViT', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Test Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Training Accuracy
    ax = axes[1, 0]
    ax.plot(epochs, results['MLP']['train_accs'], 'b-', label='MLP', linewidth=2)
    ax.plot(epochs, results['CNN']['train_accs'], 'g-', label='CNN', linewidth=2)
    ax.plot(epochs, results['ViT']['train_accs'], 'r-', label='ViT', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Training Accuracy', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Test Accuracy
    ax = axes[1, 1]
    ax.plot(epochs, results['MLP']['test_accs'], 'b-', label='MLP', linewidth=2)
    ax.plot(epochs, results['CNN']['test_accs'], 'g-', label='CNN', linewidth=2)
    ax.plot(epochs, results['ViT']['test_accs'], 'r-', label='ViT', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Test Accuracy', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
    print("Comparison plot saved as 'model_comparison.png'")
    plt.close()


def plot_metrics(mlp_model, cnn_model, vit_model, results):
    """Plot bar charts for final metrics"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    model_names = ['MLP', 'CNN', 'ViT']
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    
    # Best Accuracy
    ax = axes[0]
    accuracies = [results['MLP']['best_acc'], results['CNN']['best_acc'], results['ViT']['best_acc']]
    bars = ax.bar(model_names, accuracies, color=colors, alpha=0.8)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Best Test Accuracy', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 100])
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Training Time
    ax = axes[1]
    times = [results['MLP']['training_time']/60, results['CNN']['training_time']/60, 
             results['ViT']['training_time']/60]
    bars = ax.bar(model_names, times, color=colors, alpha=0.8)
    ax.set_ylabel('Time (minutes)', fontsize=12)
    ax.set_title('Training Time', fontsize=14, fontweight='bold')
    for bar, time in zip(bars, times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{time:.1f}m', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Parameters
    ax = axes[2]
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    params = [count_parameters(mlp_model)/1e6, count_parameters(cnn_model)/1e6, 
              count_parameters(vit_model)/1e6]
    bars = ax.bar(model_names, params, color=colors, alpha=0.8)
    ax.set_ylabel('Parameters (Millions)', fontsize=12)
    ax.set_title('Model Size', fontsize=14, fontweight='bold')
    for bar, param in zip(bars, params):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{param:.2f}M', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('model_metrics.png', dpi=150, bbox_inches='tight')
    print("Metrics comparison saved as 'model_metrics.png'")
    plt.close()


def plot_class_accuracy(mlp_model, cnn_model, vit_model, test_loader):
    """Plot per-class accuracy comparison"""
    mlp_class_acc = get_class_accuracy(mlp_model, test_loader)
    cnn_class_acc = get_class_accuracy(cnn_model, test_loader)
    vit_class_acc = get_class_accuracy(vit_model, test_loader)
    
    x = np.arange(len(CLASSES))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    bars1 = ax.bar(x - width, mlp_class_acc, width, label='MLP', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x, cnn_class_acc, width, label='CNN', color='#2ecc71', alpha=0.8)
    bars3 = ax.bar(x + width, vit_class_acc, width, label='ViT', color='#e74c3c', alpha=0.8)
    
    ax.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Per-Class Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 100])
    
    plt.tight_layout()
    plt.savefig('per_class_accuracy.png', dpi=150, bbox_inches='tight')
    print("Per-class accuracy saved as 'per_class_accuracy.png'")
    plt.close()
    
    # Print detailed results
    print("\nDetailed Per-Class Accuracy:")
    print("=" * 70)
    print(f"{'Class':<12} {'MLP':<15} {'CNN':<15} {'ViT':<15}")
    print("-" * 70)
    
    for i, cls in enumerate(CLASSES):
        print(f"{cls:<12} {mlp_class_acc[i]:<15.2f}% {cnn_class_acc[i]:<15.2f}% {vit_class_acc[i]:<15.2f}%")
    
    print("=" * 70)


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution function"""
    print("="*70)
    print("CIFAR-10 Model Comparison: MLP vs CNN vs ViT")
    print("="*70)
    print(f"Device: {DEVICE}\n")
    
    # Load data
    train_loader, test_loader = load_data()
    
    # Create models
    print("\nCreating models...")
    mlp_model = MLP().to(DEVICE)
    cnn_model = CNN().to(DEVICE)
    vit_model = ViT().to(DEVICE)
    
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\nModel Parameters:")
    print(f"  MLP: {count_parameters(mlp_model):,}")
    print(f"  CNN: {count_parameters(cnn_model):,}")
    print(f"  ViT: {count_parameters(vit_model):,}")
    
    # Train models
    mlp_results = train_model(mlp_model, "MLP", train_loader, test_loader)
    cnn_results = train_model(cnn_model, "CNN", train_loader, test_loader)
    vit_results = train_model(vit_model, "ViT", train_loader, test_loader)
    
    results = {
        'MLP': mlp_results,
        'CNN': cnn_results,
        'ViT': vit_results
    }
    
    # Print summary
    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)
    print(f"{'Model':<10} {'Parameters':<15} {'Best Acc':<12} {'Time (min)':<12}")
    print("-" * 70)
    
    models = [mlp_model, cnn_model, vit_model]
    names = ['MLP', 'CNN', 'ViT']
    
    for name, model in zip(names, models):
        params = count_parameters(model)
        best_acc = results[name]['best_acc']
        time_taken = results[name]['training_time'] / 60
        print(f"{name:<10} {params:<15,} {best_acc:<12.2f}% {time_taken:<12.2f}")
    
    print("=" * 70)
    
    # Generate plots
    print("\nGenerating comparison plots...")
    plot_training_curves(results)
    plot_metrics(mlp_model, cnn_model, vit_model, results)
    plot_class_accuracy(mlp_model, cnn_model, vit_model, test_loader)
    
    # Save models
    print("\nSaving models...")
    torch.save(mlp_model.state_dict(), 'mlp_model.pth')
    torch.save(cnn_model.state_dict(), 'cnn_model.pth')
    torch.save(vit_model.state_dict(), 'vit_model.pth')
    print("Models saved!")
    
    print("\n" + "=" * 70)
    print("COMPARISON COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()

