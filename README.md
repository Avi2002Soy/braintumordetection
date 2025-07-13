# 🧠 Brain Tumor Detection from MRI using Deep Learning (with Grad-CAM)

This project provides a web application built with **Streamlit** that allows users to upload brain MRI scans and automatically detect the presence of a brain tumor using a **Convolutional Neural Network (CNN)**. It also includes **Grad-CAM visualizations** to interpret what the model is focusing on.

---

## 🚀 Live Demo

👉 [Click here to try the app on Streamlit Cloud](https://your-app-link.streamlit.app)  
_(replace with your deployed app URL)_

---

## 🖼️ Features

- Upload MRI scan (grayscale image)
- Deep learning-based classification (Tumor / No Tumor)
- Grad-CAM heatmap to visualize model attention
- Lightweight and easy to deploy

---

## 🧠 Model Details

- Architecture: CNN with 3 Conv2D + MaxPooling layers
- Input Shape: 224x224 (Grayscale)
- Output: Binary classification (sigmoid)
- Trained using augmented MRI images from Kaggle

---

## 📁 Dataset

Dataset used: [Brain MRI Images for Brain Tumor Detection (Kaggle)](https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection)

- Classes:
  - `yes`: Tumor Present
  - `no`: Tumor Absent

---

## 🛠️ How to Run Locally

### 1. Clone the Repo
```bash
git clone https://github.com/Avi2002Soy/braintumordetection.git
cd braintumordetection
