## Detailed Implementation of the LeNet-5 architecture in PyTorch.
## Contains historically accurate layer configs.
## Dataset used: MNIST 

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

# Hyperparameters
batch_size = 64
lr = 0.001
epochs = 50

image_transform = transforms.Compose([
    transforms.Pad(2),
    transforms.ToTensor(),
    transforms.Normalize(0,1)
])

train_dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=image_transform)
test_dataset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=image_transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

