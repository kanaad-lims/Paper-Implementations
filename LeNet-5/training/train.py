import torch

def train_model(model, train_loader, loss_function, optimizer, device):
    model.train()
    total_loss = 0.0
    
    for batch_features, batch_labels in train_loader:
        batch_features = batch_features.to(device)
        batch_labels = batch_labels.to(device)

        y_pred = model(batch_features)

        # Loss Calculation
        loss = loss_function(y_pred, batch_labels)

        # Backward Pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Average loss per batch
    return total_loss/len(train_loader)