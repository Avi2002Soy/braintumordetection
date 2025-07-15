#https://drive.google.com/file/d/1XBhsSrPACn7_3UzXUueANK_ETqRjee4e/view?usp=sharing
#file_id = "1XBhsSrPACn7_3UzXUueANK_ETqRjee4e"

import os
import gdown
import streamlit as st
import onnxruntime as ort
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

IMG_WIDTH, IMG_HEIGHT = 224, 224


@st.cache_resource
def load_onnx_model():
    model_path = "brain_tumor_model.onnx"
    if not os.path.exists(model_path):
        file_id = "1XBhsSrPACn7_3UzXUueANK_ETqRjee4e"  # Replace with your ONNX file ID
        gdown.download(f"https://drive.google.com/uc?id={file_id}", model_path, quiet=False)
    return ort.InferenceSession(model_path)


def preprocess_image(image: Image.Image):
    image = image.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = np.array(image).astype("float32") / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=(0, -1))  # shape: (1, 224, 224, 1)
    return img_array


def get_grad_cam_placeholder():
    # Placeholder for ONNX Grad-CAM — actual implementation would require intermediate outputs
    return np.zeros((IMG_WIDTH, IMG_HEIGHT))


# Load ONNX model
session = load_onnx_model()
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# --- UI Layout ---
st.set_page_config(page_title="Brain Tumor Detection", layout="centered")
st.title("🧠 Brain Tumor Detection from MRI Image (ONNX)")
st.markdown("Upload an MRI scan image (in grayscale) to detect the presence of a brain tumor.")

# --- Upload Image ---
uploaded_file = st.file_uploader("Choose an MRI Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("L")  # convert to grayscale
    st.image(image, caption="Uploaded MRI Image", use_column_width=True)

    # Preprocess
    img_array = preprocess_image(image)

    # Inference
    outputs = session.run([output_name], {input_name: img_array})
    prediction = outputs[0][0][0]  # Single scalar prediction

    # Placeholder heatmap
    heatmap = get_grad_cam_placeholder()  # Replace this if you extract ONNX intermediate activations
    heatmap_resized = cv2.resize(heatmap, (IMG_WIDTH, IMG_HEIGHT))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)

    original_image = np.uint8(np.squeeze(img_array) * 255)
    original_colored = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(original_colored, 0.6, heatmap_colored, 0.4, 0)

    # Grad-CAM Display
    st.subheader("🧠 Grad-CAM Heatmap (Placeholder)")
    st.image(overlay, caption="Model Attention (Mock)", use_column_width=True)

    # Prediction Display
    if prediction > 0.5:
        st.error(f"⚠️ Tumor Detected! (Confidence: {prediction:.2%})")
    else:
        st.success(f"✅ No Tumor Detected. (Confidence: {(1 - prediction):.2%})")
