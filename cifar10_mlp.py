"""
CIFAR-10 데이터셋 MLP 분류 실습 코드
강의 실습용으로 작성된 코드입니다.
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


# 하이퍼파라미터 설정
BATCH_SIZE = 128
LEARNING_RATE = 0.001
NUM_EPOCHS = 20
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# CIFAR-10 클래스 이름
CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck')


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (다층 퍼셉트론) 모델
    CIFAR-10 이미지를 분류하는 간단한 신경망
    """
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
        # 이미지를 1차원으로 펼치기 (flatten)
        x = x.view(x.size(0), -1)
        
        # 첫 번째 은닉층
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        # 두 번째 은닉층
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        # 출력층
        x = self.fc3(x)
        return x


def load_data():
    """CIFAR-10 데이터셋 로드"""
    print("데이터셋을 불러오는 중...")
    
    # 데이터 전처리: 텐서로 변환하고 정규화
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # 훈련 데이터셋
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    # 테스트 데이터셋
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    # 데이터 로더 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return train_loader, test_loader


def visualize_samples(loader):
    """Visualize dataset samples"""
    # Get one batch
    dataiter = iter(loader)
    images, labels = next(dataiter)
    
    # Denormalize images
    images = images / 2 + 0.5
    
    # Visualize in grid
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    fig.suptitle('CIFAR-10 Sample Images', fontsize=16, fontweight='bold')
    
    for idx, ax in enumerate(axes.flat):
        if idx < len(images):
            # Convert CHW -> HWC
            img = images[idx].numpy().transpose((1, 2, 0))
            ax.imshow(img)
            ax.set_title(CLASSES[labels[idx]], fontsize=10)
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('cifar10_samples.png', dpi=150, bbox_inches='tight')
    print("Sample images saved as 'cifar10_samples.png'")
    plt.close()


def train_one_epoch(model, train_loader, criterion, optimizer, epoch):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS}')
    for images, labels in pbar:
        # Move data to device
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress
        pbar.set_postfix({
            'loss': f'{running_loss/total:.4f}',
            'acc': f'{100*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


def evaluate(model, test_loader, criterion):
    """Evaluate model"""
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    # Variables for per-class accuracy
    class_correct = [0] * 10
    class_total = [0] * 10
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Evaluating'):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Calculate per-class accuracy
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    test_loss = test_loss / len(test_loader)
    test_acc = 100 * correct / total
    
    # Print per-class accuracy
    print("\nPer-class Accuracy:")
    for i in range(10):
        acc = 100 * class_correct[i] / class_total[i]
        print(f'  {CLASSES[i]:10s}: {acc:.2f}%')
    
    return test_loss, test_acc


def plot_training_history(train_losses, train_accs, test_losses, test_accs):
    """Visualize training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss graph
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, test_losses, 'r-', label='Test Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training History: Loss', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy graph
    ax2.plot(epochs, train_accs, 'b-', label='Train Accuracy', linewidth=2)
    ax2.plot(epochs, test_accs, 'r-', label='Test Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training History: Accuracy', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    print("Training history plot saved as 'training_history.png'")
    plt.close()


def main():
    """Main execution function"""
    print("="*60)
    print("CIFAR-10 MLP Classification Practice")
    print("="*60)
    print(f"Device: {DEVICE}\n")
    
    # 1. Load data
    train_loader, test_loader = load_data()
    
    # 2. Visualize sample images
    visualize_samples(train_loader)
    
    # 3. Create model
    print("\nCreating model...")
    model = MLP().to(DEVICE)
    print(model)
    
    # Print model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # 4. Setup loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 5. Training
    print(f"\nStarting training (Total {NUM_EPOCHS} epochs)")
    print("-"*60)
    
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    
    best_acc = 0.0
    
    for epoch in range(NUM_EPOCHS):
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, epoch
        )
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Evaluate
        test_loss, test_acc = evaluate(model, test_loader, criterion)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"  Test  - Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%")
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"  ✓ Best model saved (Accuracy: {best_acc:.2f}%)")
        
        print("-"*60)
    
    # 6. Visualize training history
    print("\nGenerating training history plot...")
    plot_training_history(train_losses, train_accs, test_losses, test_accs)
    
    # 7. Final results
    print("\n" + "="*60)
    print("Training Complete!")
    print(f"Best test accuracy: {best_acc:.2f}%")
    print("="*60)


if __name__ == "__main__":
    main()

