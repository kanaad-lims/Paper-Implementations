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
device = torch.