import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

# Check if CUDA is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load a pre-trained ResNet50 model
model = models.resnet50(pretrained=True)
model.eval()  # Set the model to evaluation mode
model.to(device)  # Move the model to the appropriate device

# Load the image
img_path = 'path_to_image.jpg'  # Make sure to change this to the path of your image
img = Image.open(img_path)

# Define the transformations
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Preprocess the image
input_tensor = preprocess(img)
input_batch = input_tensor.unsqueeze(0).to(device)  # Add a batch dimension and move to the appropriate device

# Check if a GPU is available and if not, use a CPU
with torch.no_grad():
    output = model(input_batch)

# Get the predicted label
_, predicted_idx = torch.max(output, 1)
predicted_idx = predicted_idx.item()

# Load the labels used by the pre-trained model
import json
import urllib

class_idx = json.load(urllib.request.urlopen('https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json'))
idx_to_label = [class_idx[str(k)][1] for k in range(len(class_idx))]

# Print the predicted label
print("Predicted label:", idx_to_label[predicted_idx])
