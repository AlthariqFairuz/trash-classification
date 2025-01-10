import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from datasets import load_dataset

class TrashDataset:
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = sample['image']
        label = sample['label']

        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.dataset)


def evaluate_model(model, test_loader, criterion):
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   model.eval()
   val_loss = 0.0
   correct = total = 0
   
   with torch.no_grad():
       for inputs, labels in test_loader:
           inputs, labels = inputs.to(device), labels.to(device)
           outputs = model(inputs)
           val_loss += criterion(outputs, labels).item()
           _, predicted = torch.max(outputs, 1)
           total += labels.size(0)
           correct += (predicted == labels).sum().item()
   
   return val_loss / len(test_loader), correct / total

def load_data(batch_size=32):
   transform = transforms.Compose([
       transforms.Resize((224, 224)),
       transforms.RandomHorizontalFlip(p=0.5),
       transforms.RandomRotation(degrees=15),
       transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
       transforms.ToTensor(),
       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
   ])
   
   dataset = load_dataset("garythung/trashnet", split="train")
   dataset = TrashDataset(dataset, transform=transform)
   
   total_size = len(dataset)
   train_size = int(0.7 * total_size)
   val_size = int(0.2 * total_size)
   test_size = total_size - train_size - val_size
   
   train_data, val_data, test_data = random_split(
       dataset, 
       [train_size, val_size, test_size],
       generator=torch.Generator().manual_seed(42)
   )
   
   train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
   val_loader = DataLoader(val_data, batch_size=batch_size)
   test_loader = DataLoader(test_data, batch_size=batch_size)
   
   return train_loader, val_loader, test_loader
