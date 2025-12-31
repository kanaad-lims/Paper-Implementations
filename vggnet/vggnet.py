# VGGNet Implementation in PyTorch
# Dataset used: CIFAR-10

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

#Device config.
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

#HyperParameters
batch_size = 64
lr = 0.001
epochs = 5

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    
])