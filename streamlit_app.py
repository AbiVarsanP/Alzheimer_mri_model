import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18
from PIL import Image

# === Custom ResNet for grayscale input ===
def get_grayscale_resnet18(num_classes=4):
    model = resnet18(pretrained=False)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# === Load the model ===
model = get_grayscale_resnet18()
model.load_state_dict(torch.load("alzheimer_mri_model.pth", map_location=torch.device("cpu")))
model.eval()

# === Labels ===
class_names = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']

# === Transform image ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# === Dark Theme Styling with Custom CSS ===
st.markdown("""
    <style>
    body {
        background-color: #121212;
        color: white;
    }
    .main {
        background-color: #121212;
        padding: 2rem;
    }
    .title {
        font-size: 42px;
        font-weight: bold;
        color: #FFA101;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 18px;
        color: #BBBBBB;
        margin-bottom: 2rem;
    }
    .upload-box {
        background-color: #1e1e1e;
        border: 1px solid #333;
        padding: 1.5rem;
        border-radius: 10px;
        margin-top: 20px;
    }
    .footer {
        margin-top: 3rem;
        text-align: center;
        font-size: 13px;
        color: #666666;
    }
    .prediction-box {
        margin-top: 20px;
        font-size: 24px;
        color: #00e676;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# === UI Header ===
st.markdown('<div class="title">🧠 Alzheimer MRI Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">A Deep Learning powered diagnostic tool for early identification of Alzheimer’s stages using MRI scans.</div>', unsafe_allow_html=True)

# === Upload Section ===
uploaded_file = st.file_uploader("📤 Upload a Brain MRI Image (JPG/PNG)", type=["jpg", "jpeg", "png"])

# === Process Image ===
if uploaded_file:
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption="🖼️ Uploaded MRI Image", width=300)

    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        label = class_names[predicted.item()]

    st.markdown(f'<div class="prediction-box">✅ Predicted Stage: {label}</div>', unsafe_allow_html=True)

# === Footer ===
st.markdown('<div class="footer">© 2025 Alzheimer AI • Built with ❤️ using Streamlit</div>', unsafe_allow_html=True)
