## Implementation of AlexNet architecture in PyTorch from scratch.
## Dataset used: MNIST

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
