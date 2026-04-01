import torch
from torchvision import transforms, models
from PIL import Image

# Load model
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)

# Modify classifier (same as training)
model.classifier[1] = torch.nn.Linear(model.last_channel, 2)

model.load_state_dict(torch.load("model.pth"))
model.eval()

# Class names (VERY IMPORTANT - same order)
classes = ['cat', 'dog']

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load image
img = Image.open("dataset/val/dog/image copy 3.png")  # put your test image here
img = transform(img).unsqueeze(0)

# Prediction
with torch.no_grad():
    output = model(img)
    _, predicted = torch.max(output, 1)

print("Prediction:", classes[predicted.item()])