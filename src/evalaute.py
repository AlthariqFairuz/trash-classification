import torch
from model import CNN
from utils import load_data
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import wandb

def test_model(model_path, test_loader, criterion, device):
    model = CNN()
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    
    test_loss = 0.0
    correct = total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            test_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Test Set Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    wandb.log({"test_confusion_matrix": wandb.Image(plt)})
    plt.close()
    
    test_acc = correct / total
    test_loss = test_loss / len(test_loader)
    return test_loss, test_acc

if __name__ == "__main__":
    wandb.init(
        project="cnn-trash-classicifations",

        config={
        "batch_size" : 32,
        "learning_rate": 0.0001,
        "architecture": "CNN",
        "epochs": 10,
        "num_classes": 6,
        }
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader = load_data()
    criterion = torch.nn.CrossEntropyLoss()
    
    test_loss, test_acc = test_model('trained-model/best_model.pth', test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
