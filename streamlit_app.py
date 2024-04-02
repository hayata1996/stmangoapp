import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import torch.nn.functional as F

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

with st.sidebar:
        st.image('mg.png')
        st.title(":sushi:健康マンゴー診断:doughnut:")
        st.subheader("葉の画像をアップロードすると、病気の種類と確率が表示されます。Resnetを転移学習させ、フロントエンドはStreamlitを使用しています。問題は病気の葉の画像がすぐに見つからないことです...")
        st.subheader("Github: https://github.com/hayata1996/mangodetectkai")
        st.subheader("トレーニングに使用した画像リンク: https://www.kaggle.com/datasets/aryashah2k/mango-leaf-disease-dataset?ref=blog.streamlit.io")


@st.cache_resource
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
        probabilities = F.softmax(output, dim=1)

    
    return output, probabilities

if file is not None:
    st.image(file, use_column_width=True)
    predictions, probabilities = import_and_predict(file, model)
    _, predicted_class = torch.max(predictions, 1)
    predicted_class = predicted_class.item()
    predicted_probability = probabilities.max(dim=1).values.item()

    class_names = ['Anthracnose', 'Bacterial Canker', 'Cutting Weevil', 'Die Back', 'Gall Midge', 'Healthy', 'Powdery Mildew', 'Sooty Mould']
    string = "Detected Disease: " + class_names[predicted_class]
    st.sidebar.error(f"Detected Disease: {string}")

  #To show the probability of the classtaling above.
    st.sidebar.success(f"Probability: {predicted_probability:.2f}")

    # The remedy suggestions can be handled similarly based on the `predicted_class`
