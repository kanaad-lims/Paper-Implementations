import torch
import torch.nn as nn
import torch.optim as optim
from Dataset.mnist import get_dataset
from model.lenet import ConvNet
from training.train import train_model
from training.eval import evaluate_model
from utils.config import Config

def main():
    # Setting device
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    # Loading config
    config = Config()

    # Loading dataset
    train_loader, test_loader = get_dataset(config.batch_size)

    # Init model, loss fn, optimizer
    model = ConvNet().to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), config.lr)

    # training loop
    for epoch in range(config.epochs):
       loss = train_model(model, train_loader, loss_function, optimizer, device)

       if epoch % 10 == 0:
           print("Epoch: {epoch}, Loss: {loss:.4f}")
    
    print("Model trained succeessfully\n")

    test_accuracy = evaluate_model(model, test_loader, device)
    train_accuracy = evaluate_model(model, train_loader, device)

    print(f"Test Accuracy: {test_accuracy:.4f}%")
    print(f"Train Accuracy: {train_accuracy:.4f}%")

if __name__ == "__main__":
    main()