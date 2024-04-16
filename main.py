## 인공신경망과 딥러닝 HW1
## 기계정보공학과 24510091 안정민

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from dataset import MNIST
from model import LeNet5, CustomMLP, LeNet5re1, LeNet5re2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.data import Subset

# GPU 확인 및 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_dir = 'C:/Users/USER/Desktop/mnist-classification/data/train/train'
mnist_dataset_train = MNIST(data_dir)

mnist_valid_dataset = Subset(mnist_dataset_train, torch.arange(10000))
mnist_train_dataset = Subset(mnist_dataset_train, torch.arange(10000, len(mnist_dataset_train)))

print("데이터셋의 길이:", len(mnist_dataset_train))
sample_idx = 0
sample_img, sample_label = mnist_dataset_train[sample_idx]
print("샘플 이미지 형태:", sample_img.shape)
print("샘플 라벨:", sample_label)

data_dir = 'C:/Users/USER/Desktop/mnist-classification/data/test'
mnist_dataset_test = MNIST(data_dir)

print("데이터셋의 길이:", len(mnist_dataset_test))
sample_idx = 0
sample_img, sample_label = mnist_dataset_test[sample_idx]
print("샘플 이미지 형태:", sample_img.shape)
print("샘플 라벨:", sample_label)



def train(model, trn_loader, device, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in trn_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    trn_loss = running_loss / len(trn_loader)
    acc = 100.0 * correct / total

    return trn_loss, acc

def test(model, tst_loader, device, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tst_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    tst_loss = running_loss / len(tst_loader)
    acc = 100.0 * correct / total

    return tst_loss, acc

def main():
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load datasets
    mnist_dataset_train = datasets.MNIST('data', train=True, download=True, transform=transform)
    
    mnist_valid_dataset = Subset(mnist_dataset_train, torch.arange(10000))
    mnist_train_dataset = Subset(mnist_dataset_train, torch.arange(10000, len(mnist_dataset_train)))

    # Create DataLoaders
    trn_loader = DataLoader(mnist_train_dataset, batch_size=64, shuffle=True)
    valid_loader = DataLoader(mnist_valid_dataset, batch_size=1000, shuffle=False)

    # Define models
    lenet5 = LeNet5().to(device)
    custom_mlp = CustomMLP().to(device)

    # Define optimizer and criterion
    optimizer_lenet5 = optim.SGD(lenet5.parameters(), lr=0.01, momentum=0.9)
    optimizer_custom_mlp = optim.SGD(custom_mlp.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss().to(device)

    # Training loop
    epochs = 10
    lenet5_stats = {'train_loss': [], 'train_acc': [], 'valid_loss': [], 'valid_acc': []}
    custom_mlp_stats = {'train_loss': [], 'train_acc': [], 'valid_loss': [], 'valid_acc': []}

    for epoch in range(1, epochs + 1):
        # Training
        lenet5_trn_loss, lenet5_acc = train(lenet5, trn_loader, device, criterion, optimizer_lenet5)
        custom_mlp_trn_loss, custom_mlp_acc = train(custom_mlp, trn_loader, device, criterion, optimizer_custom_mlp)

        # Validation
        lenet5_valid_loss, lenet5_valid_acc = test(lenet5, valid_loader, device, criterion)
        custom_mlp_valid_loss, custom_mlp_valid_acc = test(custom_mlp, valid_loader, device, criterion)

        # Logging and saving stats
        lenet5_stats['train_loss'].append(lenet5_trn_loss)
        lenet5_stats['train_acc'].append(lenet5_acc)
        lenet5_stats['valid_loss'].append(lenet5_valid_loss)
        lenet5_stats['valid_acc'].append(lenet5_valid_acc)

        custom_mlp_stats['train_loss'].append(custom_mlp_trn_loss)
        custom_mlp_stats['train_acc'].append(custom_mlp_acc)
        custom_mlp_stats['valid_loss'].append(custom_mlp_valid_loss)
        custom_mlp_stats['valid_acc'].append(custom_mlp_valid_acc)

        print(f"Epoch {epoch}:")
        print(f"LeNet-5 Train Loss: {lenet5_trn_loss:.4f}, Accuracy: {lenet5_acc:.2f}% | Validation Loss: {lenet5_valid_loss:.4f}, Accuracy: {lenet5_valid_acc:.2f}%")
        print(f"Custom MLP Train Loss: {custom_mlp_trn_loss:.4f}, Accuracy: {custom_mlp_acc:.2f}% | Validation Loss: {custom_mlp_valid_loss:.4f}, Accuracy: {custom_mlp_valid_acc:.2f}%")
        print()
        
        
    
    # Save model weights
    torch.save(lenet5.state_dict(), 'lenet5_model.pth')
    torch.save(custom_mlp.state_dict(), 'custom_mlp_model.pth')    
    
    # Convert to DataFrame
    lenet5_df = pd.DataFrame(lenet5_stats)
    custom_mlp_df = pd.DataFrame(custom_mlp_stats)
    
    # Save to CSV
    lenet5_df.to_csv('lenet5_stats.csv', index=False)
    custom_mlp_df.to_csv('custom_mlp_stats.csv', index=False)

if __name__ == '__main__':
    main()
    

# Load the statistics
lenet5_df = pd.read_csv('lenet5_stats.csv')
custom_mlp_df = pd.read_csv('custom_mlp_stats.csv')

# Plot training loss
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(lenet5_df['train_loss'], label='train', marker='o')
plt.plot(lenet5_df['valid_loss'], label='valid', marker='x')
plt.title('LeNet5 Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot training loss
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(lenet5_df['train_acc'], label='train', marker='o')
plt.plot(lenet5_df['valid_acc'], label='valid', marker='x')
plt.title('LeNet5 Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot training loss
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(custom_mlp_df['train_loss'], label='train', marker='o')
plt.plot(custom_mlp_df['valid_loss'], label='valid', marker='x')
plt.title('custom_MLP Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


# Plot training loss
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(custom_mlp_df['train_acc'], label='train', marker='o')
plt.plot(custom_mlp_df['valid_acc'], label='valid', marker='x')
plt.title('custom_MLP Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

##############################################################################
## test 데이터로 학습 평가

# 데이터 변환 및 데이터로더 설정
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


# MNIST 테스트 데이터셋 및 데이터로더 설정
test_loader = DataLoader(mnist_dataset_test, batch_size=64, shuffle=False)


# LeNet-5 모델 초기화 및 가중치 로딩
lenet5 = LeNet5(num_classes=10).to(device)
lenet5.load_state_dict(torch.load('lenet5_model.pth', map_location=device))

# CustomMLP 모델 초기화 및 가중치 로딩
custom_mlp = CustomMLP(num_classes=10).to(device)
custom_mlp.load_state_dict(torch.load('custom_mlp_model.pth', map_location=device))

# 손실 함수 설정
criterion = nn.CrossEntropyLoss()

# LeNet-5 모델 평가
lenet5_tst_loss, lenet5_acc = test(lenet5, test_loader, device, criterion)
print(f'LeNet-5 Test Loss: {lenet5_tst_loss:.4f}')
print(f'LeNet-5 Test Accuracy: {lenet5_acc:.2f}%')

# CustomMLP 모델 평가
custom_mlp_tst_loss, custom_mlp_acc = test(custom_mlp, test_loader, device, criterion)
print(f'Custom MLP Test Loss: {custom_mlp_tst_loss:.4f}')
print(f'Custom MLP Test Accuracy: {custom_mlp_acc:.2f}%')


# =============================================================================
# LeNet-5 Test Loss: 0.0378
# LeNet-5 Test Accuracy: 98.85%
# Custom MLP Test Loss: 0.0874
# Custom MLP Test Accuracy: 97.61%
# =============================================================================





################################################################################
## LeNet5 모델에 정규화 기술을 적용하기 위해 Batch Normalization을 추가 = LeNet5re1
## LeNet5과 LeNet5re1 모델 비교

def main():
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load datasets
    mnist_dataset_train = datasets.MNIST('data', train=True, download=True, transform=transform)
    
    mnist_valid_dataset = Subset(mnist_dataset_train, torch.arange(10000))
    mnist_train_dataset = Subset(mnist_dataset_train, torch.arange(10000, len(mnist_dataset_train)))

    # Create DataLoaders
    trn_loader = DataLoader(mnist_train_dataset, batch_size=64, shuffle=True)
    valid_loader = DataLoader(mnist_valid_dataset, batch_size=1000, shuffle=False)

    # Define models
    lenet5re1 = LeNet5re1().to(device)
    lenet5 = LeNet5().to(device)

    # Define optimizer and criterion
    optimizer_lenet5re1 = optim.SGD(lenet5re1.parameters(), lr=0.01, momentum=0.9)
    optimizer_lenet5 = optim.SGD(lenet5.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss().to(device)

    # Training loop
    epochs = 10
    lenet5re1_stats = {'train_loss': [], 'train_acc': [], 'valid_loss': [], 'valid_acc': []}
    lenet5_stats = {'train_loss': [], 'train_acc': [], 'valid_loss': [], 'valid_acc': []}

    for epoch in range(1, epochs + 1):
        # Training and validation for LeNet5re1
        lenet5re1_trn_loss, lenet5re1_acc = train(lenet5re1, trn_loader, device, criterion, optimizer_lenet5re1)
        lenet5re1_valid_loss, lenet5re1_valid_acc = test(lenet5re1, valid_loader, device, criterion)
        
        lenet5re1_stats['train_loss'].append(lenet5re1_trn_loss)
        lenet5re1_stats['train_acc'].append(lenet5re1_acc)
        lenet5re1_stats['valid_loss'].append(lenet5re1_valid_loss)
        lenet5re1_stats['valid_acc'].append(lenet5re1_valid_acc)

        # Training and validation for LeNet5
        lenet5_trn_loss, lenet5_acc = train(lenet5, trn_loader, device, criterion, optimizer_lenet5)
        lenet5_valid_loss, lenet5_valid_acc = test(lenet5, valid_loader, device, criterion)
        
        lenet5_stats['train_loss'].append(lenet5_trn_loss)
        lenet5_stats['train_acc'].append(lenet5_acc)
        lenet5_stats['valid_loss'].append(lenet5_valid_loss)
        lenet5_stats['valid_acc'].append(lenet5_valid_acc)

        # Print epoch statistics
        print(f"Epoch {epoch}:")
        print(f"LeNet5re1 Train Loss: {lenet5re1_trn_loss:.4f}, Accuracy: {lenet5re1_acc:.2f}% | Validation Loss: {lenet5re1_valid_loss:.4f}, Accuracy: {lenet5re1_valid_acc:.2f}%")
        print(f"LeNet5 Train Loss: {lenet5_trn_loss:.4f}, Accuracy: {lenet5_acc:.2f}% | Validation Loss: {lenet5_valid_loss:.4f}, Accuracy: {lenet5_valid_acc:.2f}%")
        print()
        
    # Save model weights
    torch.save(lenet5re1.state_dict(), 'lenet5re1_model.pth')
    torch.save(lenet5.state_dict(), 'lenet5_model.pth')    
    
    # Convert to DataFrame
    lenet5re1_df = pd.DataFrame(lenet5re1_stats)
    lenet5_df = pd.DataFrame(lenet5_stats)
    
    # Save to CSV
    lenet5re1_df.to_csv('lenet5re1_stats.csv', index=False)
    lenet5_df.to_csv('lenet5_stats.csv', index=False)

if __name__ == '__main__':
    main()
    
    
# Load the statistics
lenet5re1_df = pd.read_csv('lenet5re1_stats.csv')
lenet5_df = pd.read_csv('lenet5_stats.csv')

# Plot LeNet5re1 and LeNet5 training and validation loss
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(lenet5re1_df['train_loss'], label='LeNet5re1 train', marker='o')
plt.plot(lenet5re1_df['valid_loss'], label='LeNet5re1 valid', marker='x')
plt.plot(lenet5_df['train_loss'], label='LeNet5 train', marker='s')
plt.plot(lenet5_df['valid_loss'], label='LeNet5 valid', marker='d')
plt.title('LeNet5re1 vs LeNet5 Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot LeNet5re1 and LeNet5 training and validation accuracy
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(lenet5re1_df['train_acc'], label='LeNet5re1 train', marker='o')
plt.plot(lenet5re1_df['valid_acc'], label='LeNet5re1 valid', marker='x')
plt.plot(lenet5_df['train_acc'], label='LeNet5 train', marker='s')
plt.plot(lenet5_df['valid_acc'], label='LeNet5 valid', marker='d')
plt.title('LeNet5re1 vs LeNet5 Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# MNIST test dataset and dataloader setup
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_loader = DataLoader(mnist_dataset_test, batch_size=64, shuffle=False)


# Initialize LeNet-5re1 and load weights
lenet5re1 = LeNet5re1(num_classes=10).to(device)
lenet5re1.load_state_dict(torch.load('lenet5re1_model.pth', map_location=device))

lenet5 = LeNet5(num_classes=10).to(device)
lenet5.load_state_dict(torch.load('lenet5_model.pth', map_location=device))

# Loss function setup
criterion = nn.CrossEntropyLoss()

# Evaluate LeNet-5re1 model
lenet5re1_tst_loss, lenet5re1_acc = test(lenet5re1, test_loader, device, criterion)
print(f'LeNet-5re1 Test Loss: {lenet5re1_tst_loss:.4f}')
print(f'LeNet-5re1 Test Accuracy: {lenet5re1_acc:.2f}%')

# Evaluate LeNet-5 model
lenet5_tst_loss, lenet5_acc = test(lenet5, test_loader, device, criterion)
print(f'LeNet-5 Test Loss: {lenet5_tst_loss:.4f}')
print(f'LeNet-5 Test Accuracy: {lenet5_acc:.2f}%')

# =============================================================================
# LeNet-5re1 Test Loss: 0.0337
# LeNet-5re1 Test Accuracy: 99.01%
# LeNet-5 Test Loss: 0.0414
# LeNet-5 Test Accuracy: 98.95%
# =============================================================================



################################################################################
## LeNet5 모델에 정규화 기술을 적용하기 위해 Batch Normalization과 dropout을 추가 = LeNet5re2
## LeNet5과 LeNet5re2 모델 비교

def main():
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load datasets
    mnist_dataset_train = datasets.MNIST('data', train=True, download=True, transform=transform)
    
    mnist_valid_dataset = Subset(mnist_dataset_train, torch.arange(10000))
    mnist_train_dataset = Subset(mnist_dataset_train, torch.arange(10000, len(mnist_dataset_train)))

    # Create DataLoaders
    trn_loader = DataLoader(mnist_train_dataset, batch_size=64, shuffle=True)
    valid_loader = DataLoader(mnist_valid_dataset, batch_size=1000, shuffle=False)

    # Define models
    lenet5re2 = LeNet5re2().to(device)
    lenet5 = LeNet5().to(device)

    # Define optimizer and criterion
    optimizer_lenet5re2 = optim.SGD(lenet5re2.parameters(), lr=0.01, momentum=0.9)
    optimizer_lenet5 = optim.SGD(lenet5.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss().to(device)

    # Training loop
    epochs = 10
    lenet5re2_stats = {'train_loss': [], 'train_acc': [], 'valid_loss': [], 'valid_acc': []}
    lenet5_stats = {'train_loss': [], 'train_acc': [], 'valid_loss': [], 'valid_acc': []}

    for epoch in range(1, epochs + 1):
        # Training and validation for lenet5re2
        lenet5re2_trn_loss, lenet5re2_acc = train(lenet5re2, trn_loader, device, criterion, optimizer_lenet5re2)
        lenet5re2_valid_loss, lenet5re2_valid_acc = test(lenet5re2, valid_loader, device, criterion)
        
        lenet5re2_stats['train_loss'].append(lenet5re2_trn_loss)
        lenet5re2_stats['train_acc'].append(lenet5re2_acc)
        lenet5re2_stats['valid_loss'].append(lenet5re2_valid_loss)
        lenet5re2_stats['valid_acc'].append(lenet5re2_valid_acc)

        # Training and validation for LeNet5
        lenet5_trn_loss, lenet5_acc = train(lenet5, trn_loader, device, criterion, optimizer_lenet5)
        lenet5_valid_loss, lenet5_valid_acc = test(lenet5, valid_loader, device, criterion)
        
        lenet5_stats['train_loss'].append(lenet5_trn_loss)
        lenet5_stats['train_acc'].append(lenet5_acc)
        lenet5_stats['valid_loss'].append(lenet5_valid_loss)
        lenet5_stats['valid_acc'].append(lenet5_valid_acc)

        # Print epoch statistics
        print(f"Epoch {epoch}:")
        print(f"lenet5re2 Train Loss: {lenet5re2_trn_loss:.4f}, Accuracy: {lenet5re2_acc:.2f}% | Validation Loss: {lenet5re2_valid_loss:.4f}, Accuracy: {lenet5re2_valid_acc:.2f}%")
        print(f"LeNet5 Train Loss: {lenet5_trn_loss:.4f}, Accuracy: {lenet5_acc:.2f}% | Validation Loss: {lenet5_valid_loss:.4f}, Accuracy: {lenet5_valid_acc:.2f}%")
        print()
        
    # Save model weights
    torch.save(lenet5re2.state_dict(), 'lenet5re2_model.pth')
    torch.save(lenet5.state_dict(), 'lenet5_model.pth')    
    
    # Convert to DataFrame
    lenet5re2_df = pd.DataFrame(lenet5re2_stats)
    lenet5_df = pd.DataFrame(lenet5_stats)
    
    # Save to CSV
    lenet5re2_df.to_csv('lenet5re2_stats.csv', index=False)
    lenet5_df.to_csv('lenet5_stats.csv', index=False)

if __name__ == '__main__':
    main()
    
    
# Load the statistics
lenet5re2_df = pd.read_csv('lenet5re2_stats.csv')
lenet5_df = pd.read_csv('lenet5_stats.csv')

# Plot lenet5re2 and LeNet5 training and validation loss
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(lenet5re2_df['train_loss'], label='LeNet5re2 train', marker='o')
plt.plot(lenet5re2_df['valid_loss'], label='LeNet5re2 valid', marker='x')
plt.plot(lenet5_df['train_loss'], label='LeNet5 train', marker='s')
plt.plot(lenet5_df['valid_loss'], label='LeNet5 valid', marker='d')
plt.title('LeNet5re2 vs LeNet5 Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot lenet5re2 and LeNet5 training and validation accuracy
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(lenet5re2_df['train_acc'], label='LeNet5re2 train', marker='o')
plt.plot(lenet5re2_df['valid_acc'], label='LeNet5re2 valid', marker='x')
plt.plot(lenet5_df['train_acc'], label='LeNet5 train', marker='s')
plt.plot(lenet5_df['valid_acc'], label='LeNet5 valid', marker='d')
plt.title('LeNet5re2 vs LeNet5 Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# MNIST test dataset and dataloader setup
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_loader = DataLoader(mnist_dataset_test, batch_size=64, shuffle=False)


# Initialize LeNet-5re1 and load weights
lenet5re2 = LeNet5re2(num_classes=10).to(device)
lenet5re2.load_state_dict(torch.load('lenet5re2_model.pth', map_location=device))

lenet5 = LeNet5(num_classes=10).to(device)
lenet5.load_state_dict(torch.load('lenet5_model.pth', map_location=device))

# Loss function setup
criterion = nn.CrossEntropyLoss()

# Evaluate LeNet-5re1 model
lenet5re2_tst_loss, lenet5re2_acc = test(lenet5re2, test_loader, device, criterion)
print(f'LeNet-5re2 Test Loss: {lenet5re2_tst_loss:.4f}')
print(f'LeNet-5re2 Test Accuracy: {lenet5re2_acc:.2f}%')

# Evaluate LeNet-5 model
lenet5_tst_loss, lenet5_acc = test(lenet5, test_loader, device, criterion)
print(f'LeNet-5 Test Loss: {lenet5_tst_loss:.4f}')
print(f'LeNet-5 Test Accuracy: {lenet5_acc:.2f}%')

# =============================================================================
# LeNet-5re2 Test Loss: 0.0305
# LeNet-5re2 Test Accuracy: 99.09%
# LeNet-5 Test Loss: 0.0437
# LeNet-5 Test Accuracy: 98.64%
# =============================================================================
