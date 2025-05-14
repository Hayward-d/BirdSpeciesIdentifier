import torch
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torch import nn
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class BirdDataset(Dataset):
    def __init__(self, root_dir, is_train=True, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        images = pd.read_csv(os.path.join(root_dir, "images.txt"), sep=" ", names=["img_id", "img_path"])
        labels = pd.read_csv(os.path.join(root_dir, "image_class_labels.txt"), sep=" ", names=["img_id", "label"])
        split = pd.read_csv(os.path.join(root_dir, "train_test_split.txt"), sep=" ", names=["img_id", "is_train"])

        data = images.merge(labels, on="img_id").merge(split, on="img_id")
        data = data[data["is_train"] == int(is_train)]
        data["label"] -= 1  # convert to 0-based

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

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

test_dataset = BirdDataset("Data/CUB_200_2011", is_train=False, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 200)  # 200 bird species

model.load_state_dict(torch.load("bird_model.pth", map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

top1_correct = 0
top3_correct = 0
top5_correct = 0
total = 0

model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)

        # Top-1 prediction
        _, top1_pred = torch.max(outputs, 1)
        top1_correct += (top1_pred == labels).sum().item()

        # Top-3 predictions
        top3_preds = torch.topk(outputs, k=3, dim=1).indices
        for i in range(labels.size(0)):
            if labels[i] in top3_preds[i]:
                top3_correct += 1

        # Top-5 predictions
        top5_preds = torch.topk(outputs, k=5, dim=1).indices
        for i in range(labels.size(0)):
            if labels[i] in top5_preds[i]:
                top5_correct += 1

        total += labels.size(0)

top1_acc = 100 * top1_correct / total
top3_acc = 100 * top3_correct / total
top5_acc = 100 * top5_correct / total

print(f"Top-1 Accuracy: {top1_acc:.2f}%")
print(f"Top-3 Accuracy: {top3_acc:.2f}%")
print(f"Top-5 Accuracy: {top5_acc:.2f}%")

