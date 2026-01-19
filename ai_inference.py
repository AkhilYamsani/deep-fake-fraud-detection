import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Load pretrained deepfake detector (cloud-safe)
processor = AutoImageProcessor.from_pretrained(
    "dima806/deepfake_vs_real_image_detection"
)
model = AutoModelForImageClassification.from_pretrained(
    "dima806/deepfake_vs_real_image_detection"
)

model.eval()

def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")

    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)

    confidence, pred = torch.max(probs, dim=1)
    label = "Fake" if pred.item() == 1 else "Real"

    return label, float(confidence)
