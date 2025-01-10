import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from tqdm import tqdm
from model import CNN
from utils import load_data, evaluate_model
import os

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, patience):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in tqdm(range(epochs), desc='Epochs'):
        running_loss = 0.0
        correct = total = 0
        
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        val_loss, val_acc = evaluate_model(model, val_loader, criterion)
        
        wandb.log({
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "epoch": epoch + 1
        })
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            os.makedirs('trained-model', exist_ok=True)
            torch.save(model.state_dict(), 'trained-model/best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

if __name__ == "__main__":
    wandb.init(
        project="cnn-trash-classicifations",
        config={
            "batch_size": 32,
            "learning_rate": 0.0001,
            "architecture": "CNN", 
            "epochs": 10,
            "num_classes": 6,
        }
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    train_loader, val_loader, _ = load_data()
    train_model(model, train_loader, val_loader, criterion, optimizer, epochs=1, patience=3)
    wandb.finish()
