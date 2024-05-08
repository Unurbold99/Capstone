import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import joblib
import torch

# Load your trained model
model = joblib.load('x_ray_pred.pkl')

# Define the class labels for your project
class_labels = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
                  "Mass", "Nodule", "Pneumonia", "Pneumothorax",
                  "Consolidation", "No Finding"]

# Function to make predictions on uploaded image
def predict_image(image):
    img = np.array(image)
    prediction = model.predict(img)
    return prediction

# Streamlit UI
st.title('X-ray Image Classification Dashboard')

# Introduction Section
st.write('Welcome to the X-ray Image Classification Dashboard.')
st.write('This application showcases a deep learning model for classifying X-ray images into different categories of diseases. The data was takeng from url: https://www.kaggle.com/datasets/nih-chest-xrays/data/data. The dataset has 14 different classes, with over 112,000 x-ray images.')

# Upload Image Section
st.sidebar.title('Upload X-ray Image')
uploaded_file = st.sidebar.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img = np.array(image)  # Convert PIL Image to numpy array
    st.image(img, caption='Uploaded Image', use_column_width=True)

    if st.sidebar.button('Classify'):
        with st.spinner('Classifying...'):
            prediction = predict_image(img)  # Pass the numpy array to the predict_image function
            st.write('**Prediction Results:**')
            for i in range(len(class_labels)):
                st.write(f'- {class_labels[i]}: {prediction[0][i]*100:.2f}%')

# About Section
st.sidebar.title('About')
st.sidebar.info('This dashboard was created by [Your Name].\n'
                'It is meant for demonstration purposes only, and should not be taken as a medical advisory')

# Model Architecture Section
st.sidebar.title('Model Architecture')
st.sidebar.image('model_architecture.png', caption='Model Architecture', use_column_width=True)
