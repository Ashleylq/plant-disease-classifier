import torchvision.models as models
import torch.nn as nn
from torchvision.transforms import v2 as transforms
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import gradio as gr
import torch


model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 38)
path = hf_hub_download(repo_id="imightlikelemonade-97/plantvillage-resnet50", filename="model.safetensors")
state_dict = load_file(path)
model.load_state_dict(state_dict)
model.eval()

preprocess = transforms.Compose([
  transforms.toImage(),
  transforms.toDtype(torch.float32, scale=True),
  transforms.Resize(size=224, antialias=True),
  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_classification(inp):
  tensor = preprocess(inp).unsqueeze(0)
  with torch.no_grad():
    logits = model(tensor)
    probabilities = F.softmax(logits, dim=1)[0]
  labels = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']
  return {labels[i]: float(probabilities[i]) for i in range(len(probabilities))}


image = gr.Image(size=(224, 224))
labels = gr.Label(num_top_classes=3)
gr.Interface(fn=get_classification, inputs=image, outputs=labels)