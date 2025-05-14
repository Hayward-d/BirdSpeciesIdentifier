import os
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms

#THIS PYTHON FILE CREATES, DEFINES, EVALUATES, AND SAVES THE RESNET18 CNN MODEL

# CREATE DATASET
class BirdDataset(Dataset):
    def __init__(self, root_dir, is_train=True, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        images = pd.read_csv(os.path.join(root_dir, "images.txt"), sep=" ", names=["img_id", "img_path"])
        labels = pd.read_csv(os.path.join(root_dir, "image_class_labels.txt"), sep=" ", names=["img_id", "label"])
        split = pd.read_csv(os.path.join(root_dir, "train_test_split.txt"), sep=" ", names=["img_id", "is_train"])

        data = images.merge(labels, on="img_id").merge(split, on="img_id")
        data = data[data["is_train"] == int(is_train)]

        data["label"] -= 1

        self.data = data.reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, "images", self.data.loc[idx, "img_path"])
        image = Image.open(img_path).convert("RGB")
        label = self.data.loc[idx, "label"]

        if self.transform:
            image = self.transform(image)

        return image, label


# DEFINE TRANSFORM & LOADERS
import torch
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = BirdDataset("Data/CUB_200_2011", is_train=True, transform=transform)
test_dataset = BirdDataset("Data/CUB_200_2011", is_train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# DEFINING MODEL
from torchvision import models
import torch.nn as nn

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 200)

# TRAINING MODEL
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader):.4f}")

# EVALUATING MODEL
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")

# SAVE MODEL
torch.save(model.state_dict(), "bird_model.pth")
