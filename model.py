## 인공신경망과 딥러닝 HW1
## 기계정보공학과 24510091 안정민

import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    """ LeNet-5 (LeCun et al., 1998)

        - For a detailed architecture, refer to the lecture note
        - Freely choose activation functions as you want
        - For subsampling, use max pooling with kernel_size = (2,2)
        - Output should be a logit vector
    """

    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, img):
        x = nn.functional.relu(self.conv1(img))
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(-1, 16*4*4)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    
class CustomMLP(nn.Module):
    """ 사용자 정의 다층 퍼셉트론(MLP) 모델

        - 모델 파라미터 수는 LeNet-5와 비슷하게 유지되어야 합니다.
    """

    def __init__(self, num_classes=10):
        super(CustomMLP, self).__init__()
        self.fc1 = nn.Linear(28*28, 70)
        self.fc2 = nn.Linear(70, 50)
        self.fc3 = nn.Linear(50, num_classes)

    def forward(self, img):
        x = img.view(-1, 28*28)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    
## LeNet5 모델에 정규화 기술을 적용하기 위해 Batch Normalization을 추가 = LeNet5re1
## Batch Normalization : 각 레이어의 입력을 평균과 분산으로 정규화하여 학습을 안정화시키고,
## 수렴 속도를 빠르게 하는 데 도움

class LeNet5re1(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5re1, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, img):
        x = nn.functional.relu(self.bn1(self.conv1(img)))
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = nn.functional.relu(self.bn2(self.conv2(x)))
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(-1, 16*4*4)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x   
    
    
## Batch Normalization과 Dropout을 추가하여 더 정규화된 모델 = LeNet5re2

class LeNet5re2(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5re2, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        
        # Dropout layers
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)

    def forward(self, img):
        x = F.relu(self.bn1(self.conv1(img)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(-1, 16*4*4)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)  # Adding dropout after FC1
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)  # Adding dropout after FC2
        x = self.fc3(x)
        return x