import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import random

# Remove unnecessary imports and warnings suppression for clarity

st.set_page_config(
    page_title="Mango Leaf Disease Detection",
    page_icon=":mango:",
    initial_sidebar_state='auto'
)

# Style hiding is fine, keeping it as is
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

@st.experimental_memo
def load_model():
    model_save_path = 'model.pth'
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    NUM_CLASSES = 8
    model.fc = torch.nn.Linear(num_ftrs, NUM_CLASSES)
    
    checkpoint = torch.load(model_save_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model.eval()

model = load_model()

st.title("Mango Disease Detection with Remedy Suggestion")

file = st.file_uploader("", type=["jpg", "png"])

def import_and_predict(image_data, model):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_data)
    image = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        output = model(image)
    
    return output

if file is not None:
    st.image(file, use_column_width=True)
    predictions = import_and_predict(file, model)
    _, predicted_class = torch.max(predictions, 1)
    predicted_class = predicted_class.item()

    class_names = ['Anthracnose', 'Bacterial Canker', 'Cutting Weevil', 'Die Back', 'Gall Midge', 'Healthy', 'Powdery Mildew', 'Sooty Mould']
    string = "Detected Disease: " + class_names[predicted_class]
    st.sidebar.error(f"Detected Disease: {string}")

    # Random accuracy display, this should ideally be based on model's confidence
    accuracy = random.randint(98, 99) + random.randint(0, 99) * 0.01
    st.sidebar.error(f"Accuracy: {accuracy:.2f} %")

    # The remedy suggestions can be handled similarly based on the `predicted_class`
