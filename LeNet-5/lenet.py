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
lr = 0.01
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

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.tanh(self.conv1(x)))
        x = self.pool(torch.tanh(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x
    

model = ConvNet().to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)


n_total_steps = len(train_loader)
for epoch in range(epochs):
    total_epoch_loss = 0

    for batch_features, batch_labels in train_loader:
        batch_features = batch_features.to(device)
        batch_labels = batch_labels.to(device)

        #Forward Pass
        y_pred = model(batch_features)

        #Loss Calculation
        loss = loss_function(y_pred, batch_labels)

        #Backward Pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_epoch_loss += loss.item()

    avg_loss = total_epoch_loss/len(train_loader)
    if(epoch%10 == 0):
        print(f"Epoch: {epoch} Loss: {avg_loss}")
print("Model trained successfully\n")


#Evaluation
# Setting the model into evaluation mode.
print(model.eval())
model.eval()
total = 0
correct = 0

with torch.no_grad():

    for batch_features, batch_labels in test_loader:
        batch_features = batch_features.to(device)
        batch_labels = batch_labels.to(device)
        output = model(batch_features)
        

        _, predicted = torch.max(output, 1) #Extracting the label for the maximum probabilistic value returned by the model.
        
        total = total + batch_labels.shape[0]
        correct = correct + (predicted == batch_labels).sum().item()
    

accuracy = correct/total
print(f"Test Accuracy: {accuracy * 100:.4f}%")



total = 0
correct = 0

with torch.no_grad():

    for batch_features, batch_labels in train_loader:
        batch_features = batch_features.to(device)
        batch_labels = batch_labels.to(device)
        output = model(batch_features)
        

        _, predicted = torch.max(output, 1) #Extracting the label for the maximum probabilistic value returned by the model.
        
        total = total + batch_labels.shape[0]
        correct = correct + (predicted == batch_labels).sum().item()
    

accuracy = correct/total
print(f"Train Accuracy: {accuracy * 100:.4f}%")