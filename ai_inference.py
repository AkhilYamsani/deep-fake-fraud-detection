import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import timm

# Load pretrained deepfake model
model = timm.create_model(
    "xception",
    pretrained=False,
    num_classes=2
)

# Load FaceForensics++ weights
state_dict = torch.hub.load_state_dict_from_url(
    "https://github.com/selimsef/dfdc_deepfake_challenge/releases/download/v1.0/xception_ffpp.pth",
    map_location="cpu"
)

model.load_state_dict(state_dict)
model.eval()

transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        probs = F.softmax(output, dim=1)

    confidence, pred = torch.max(probs, 1)

    label = "Fake" if pred.item() == 1 else "Real"
    return label, float(confidence)
