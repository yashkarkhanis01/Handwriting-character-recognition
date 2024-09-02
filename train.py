# train.py
import torch
import torch.nn.functional as F

def train_step(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer, device: torch.device):
    model.train()
    train_loss = 0.0 
    
    for X, _ in dataloader:
        X = X.to(device)
        
        batch_size, channels, height, width = X.size()
        X = X.view(batch_size, -1) # Flattening Image
        X = X.unsqueeze(1) # Adding Sequence Dimension
        
        hidden = model.init_hidden(X.size(0), device)
        
        optimizer.zero_grad()
        
        outputs, _ = model(X, hidden)
        
        outputs = outputs.squeeze(1) 
        targets = X.view(X.size(0), -1) # Flattening the targets
        
        loss = loss_fn(outputs, targets)
        
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
    return train_loss / len(dataloader)

def test_step(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module, device: torch.device):
    model.eval()
    test_loss = 0.0 
    with torch.inference_mode():
        for X, _ in dataloader:
            X = X.to(device)
            
            batch_size, channels, height, width = X.size()
            X = X.view(batch_size, -1)
            X = X.unsqueeze(1)
            
            hidden = model.init_hidden(X.size(0), device)
            
            outputs, _ = model(X, hidden)

            outputs = outputs.squeeze(1)
            targets = X.view(X.size(0), -1)

            loss = loss_fn(outputs, targets)
            test_loss += loss.item()
    return test_loss / len(dataloader)

def train(model: torch.nn.Module, train_dataloader: torch.utils.data.DataLoader, val_dataloader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, loss_fn: torch.nn.Module, device: torch.device, epochs: int = 5):
    model.to(device)
    results = {"train_loss": [], "val_loss": []}
    
    for epoch in range(epochs):
        train_loss = train_step(model, train_dataloader, loss_fn, optimizer, device)
        val_loss = test_step(model, val_dataloader, loss_fn, device)
        
        print(f"Epoch: {epoch + 1} | train_loss: {train_loss:.3f} | val_loss: {val_loss:.3f}")
        results["train_loss"].append(train_loss)
        results["val_loss"].append(val_loss)
    
    return results
