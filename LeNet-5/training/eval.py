import torch

def evaluate_model(model, test_loader, device):
    model.eval()
    total, correct = 0, 0

    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            output = model(batch_features)
            _, predicted = torch.max(output, 1) # Extracting the label for the maximum probabilistic value returned by the model.

            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()

            accuracy = correct / total
            accuracy = accuracy * 100
        
        return accuracy