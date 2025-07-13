#https://drive.google.com/file/d/1vTShYm_6a4edn5H7sblMlpYvLOap9Naa/view?usp=sharing
#file_id = "1vTShYm_6a4edn5H7sblMlpYvLOap9Naa"

import os
import gdown
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import io

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
IMG_WIDTH, IMG_HEIGHT = 224, 224


@st.cache_resource
def load_model_from_drive():
    model_path = "brain_tumor_model.h5"
    if not os.path.exists(model_path):
        file_id = "1vTShYm_6a4edn5H7sblMlpYvLOap9Naa"  # replace with your ID
        gdown.download(f"https://drive.google.com/uc?id={file_id}", model_path, quiet=False)
    return load_model(model_path)


def get_grad_cam(model, image_array, layer_name="conv2d_2"):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()
    return heatmap

# Load the pre-trained model
model = load_model_from_drive()

# --- APP UI ---
st.set_page_config(page_title="Brain Tumor Detection", layout="centered")
st.title("🧠 Brain Tumor Detection from MRI Image")
st.markdown("Upload an MRI scan image (in grayscale) to detect the presence of a brain tumor.")

# --- IMAGE UPLOAD ---
uploaded_file = st.file_uploader("Choose an MRI Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Show original image
    image = Image.open(uploaded_file).convert("L")  # convert to grayscale
    st.image(image, caption="Uploaded MRI Image", use_column_width=True)

    # Preprocess the image
    img = image.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dim
    img_array = np.expand_dims(img_array, axis=-1)  # Add channel dim: (1, 224, 224, 1)

    # Predict
    # Make prediction and generate Grad-CAM
    prediction = model.predict(img_array)[0][0]
    heatmap = get_grad_cam(model, img_array)
    # # Process heatmap
    heatmap_resized = cv2.resize(heatmap, (IMG_WIDTH, IMG_HEIGHT))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    # Overlay heatmap on original image
    original_image = np.uint8(np.squeeze(img_array) * 255)
    original_colored = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(original_colored, 0.6, heatmap_colored, 0.4, 0)
    # Convert to Image for display
    st.subheader("🧠 Grad-CAM Heatmap")
    st.image(overlay, caption="Model Attention (Grad-CAM)", use_column_width=True)
    # Prediction result
    if prediction > 0.5:
        st.error(f"⚠️ Tumor Detected! (Confidence: {prediction:.2%})")
    else:
        st.success(f"✅ No Tumor Detected. (Confidence: {(1 - prediction):.2%})")
    # Show result
