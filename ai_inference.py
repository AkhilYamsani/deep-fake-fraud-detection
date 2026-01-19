import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Load pretrained MobileNet
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

# Modify classifier for binary classification
model.classifier[1] = nn.Linear(model.last_channel, 2)

model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)

    confidence, predicted = torch.max(probs, 1)
    label = "Fake" if predicted.item() == 1 else "Real"

    return label, float(confidence)