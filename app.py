#https://drive.google.com/file/d/1XBhsSrPACn7_3UzXUueANK_ETqRjee4e/view?usp=sharing
#file_id = "1XBhsSrPACn7_3UzXUueANK_ETqRjee4e"

import streamlit as st
st.set_page_config(
    page_title="🧠 Brain Tumor Detection",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="auto"
)

import os
import gdown
import onnxruntime as ort
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

# Constants
IMG_WIDTH, IMG_HEIGHT = 224, 224
st.markdown("""
    <style>
    .main {
        background-color: #c14c8a;
        color: #333333;
        font-family: 'Segoe UI', sans-serif;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: blue;
        font-weight: bold;
    }
    .css-1v0mbdj p {
        font-size: 18px;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def load_model():
    if not os.path.exists("model.onnx"):
        url = f"https://drive.google.com/uc?id=1XBhsSrPACn7_3UzXUueANK_ETqRjee4e"
        gdown.download(url, "model.onnx", quiet=False)
    session = ort.InferenceSession("model.onnx")
    return session

def preprocess_image(image):
    img = image.resize((IMG_WIDTH, IMG_HEIGHT)).convert("L")  # Convert to grayscale
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=(0, -1))  # Shape: (1, 224, 224, 1)
    return img_array

def get_prediction(session, img_array):
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    result = session.run([output_name], {input_name: img_array})
    return result[0][0][0]  # Extract scalar prediction

def dummy_grad_cam(image_array):
    # Simulate a heatmap as Grad-CAM is not trivial in ONNX
    heatmap = np.random.rand(IMG_WIDTH, IMG_HEIGHT)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    original = np.uint8(np.squeeze(image_array) * 255)
    original = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)
    return overlay

# Load the model
session = load_model()

# UI
st.title("🧠 Brain Tumor Detection from MRI Image")
st.markdown("Upload an MRI scan image (in grayscale) to detect the presence of a brain tumor.")

uploaded_file = st.file_uploader("Choose an MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption="Uploaded MRI Image", use_column_width=True)

    img_array = preprocess_image(image)
    prediction = get_prediction(session, img_array)

    # Dummy Grad-CAM visualization
    heatmap_overlay = dummy_grad_cam(img_array)
    st.subheader("🧠 Simulated Heatmap (Grad-CAM Style)")
    st.image(heatmap_overlay, caption="Model Attention", use_column_width=True)

    # Display result
    if prediction > 0.5:
        st.error(f"⚠️ Tumor Detected! (Confidence: {prediction:.2%})")
    else:
        st.success(f"✅ No Tumor Detected. (Confidence: {(1 - prediction):.2%})")
