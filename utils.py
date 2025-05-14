import torch
from torchvision import models, transforms
from PIL import Image
import os

def load_model():
    # ResNet18:
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Linear(model.fc.in_features, 200)
    model.load_state_dict(torch.load("bird_model.pth", map_location="cpu"))

    # ResNet50:
    #model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    #model.fc = torch.nn.Linear(model.fc.in_features, 200)
    #model.load_state_dict(torch.load("bird_model_50.pth", map_location="cpu"))

    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    class_names = []
    class_to_folder = {}
    with open("classes.txt") as f:
        for line in f:
            idx, name = line.strip().split(" ")
            name_example = name.replace("_", " ")
            name_example = name_example[4:]
            class_names.append(name_example)
            class_to_folder[int(idx) - 1] = name

    return model, transform, class_names, class_to_folder

def predict_top3(img_path, model, transform, class_names):
    image = Image.open(img_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        top3_probs, top3_indices = torch.topk(probs, 3)

    predictions = []
    for i in range(3):
        idx = top3_indices[i].item()
        species = class_names[idx]
        example_image_path = f"static/ExamplePhotos/{species.replace(' ', '').lower()}"
        example_image_path = example_image_path + ".jpg"

        predictions.append({
            'species': species,
            'prob': f"{top3_probs[i].item()*100:.2f}%",
            'example_image': example_image_path
        })

    return predictions
