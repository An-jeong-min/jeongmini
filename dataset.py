## 인공신경망과 딥러닝 HW1
## 기계정보공학과 24510091 안정민

import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class MNIST(Dataset):
    """ MNIST dataset

        To write custom datasets, refer to
        https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

    Args:
        data_dir: directory path containing images

    Note:
        1) Each image should be preprocessed as follows:
            - First, all values should be in a range of [0,1]
            - Subtract mean of 0.1307, and divide by std 0.3081
            - These preprocessing can be implemented using torchvision.transforms
        2) Labels can be obtained from filenames: {number}_{label}.png
    """

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.image_files = [f for f in os.listdir(data_dir) if f.endswith('.png')]
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.data_dir, img_name)
        img = Image.open(img_path)
        
        # Apply transformations
        img = self.transform(img)
        
        # Extract label from filename
        label = int(img_name.split('_')[1].split('.')[0])

        return img, label

if __name__ == '__main__':
    # Test codes to verify the implementation
    data_dir = 'C:/Users/USER/Desktop/mnist-classification/data/train/train'
    mnist_dataset = MNIST(data_dir)
    
    # data_dir = 'C:/Users/USER/Desktop/mnist-classification/data/train/train'
    # mnist_dataset = MNIST(data_dir)

    # Check length of dataset
    print("Length of dataset:", len(mnist_dataset))

    # Check sample image and label
    sample_idx = 0
    sample_img, sample_label = mnist_dataset[sample_idx]
    print("Sample image shape:", sample_img.shape)
    print("Sample label:", sample_label)