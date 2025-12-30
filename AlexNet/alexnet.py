## Implementation of AlexNet architecture in PyTorch from scratch.
## Dataset used: CIFAR-10.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

#Device config.
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
batch_size = 128
lr = 0.01
epochs = 5
momentum_value = 0.9

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(0,1)
])

train_dataset = torchvision.datasets.Caltech101(root="./data", train=True, download=True, transform=image_transform)
test_dataset = torchvision.datasets.Caltech101(root="./data", train=False, download=True, transform=image_transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, 11, stride=4, padding=2)
        self.pool1 = nn.MaxPool2d(3, 2)
        self.conv2 = nn.Conv2d(96, 256, 5, 1, 2)
        self.pool2 = nn.MaxPool2d(3, 2)
        self.conv3 = nn.Conv2d(256, 384, 3, 1, 1)
        self.conv4 = nn.Conv2d(384, 384, 3, 1, 1)
        self.conv5 = nn.Conv2d(384, 256, 3, 1, 1)
        self.pool5 = nn.MaxPool2d(3, 2)
        self.fc1 = nn.Linear(256*6*6, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 101)

        # Initilization of relu and dropout functions.
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
    
    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.pool5(self.relu(self.conv5(x)))
        x = torch.flatten(x, 1) # Converted from 4D tensor to a 2D tensor (batch_size, features)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)

        return x
    

model = ConvNet().to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum_value)


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
