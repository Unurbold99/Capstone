import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import joblib
import torch

# Load the trained model
model = joblib.load('x_ray_pred.pkl')

# Define a function to make predictions
def predict(image):
    # Convert the image to a numpy array
    image = np.array(image)
    # Resize the image to the expected size
    image = cv2.resize(image, (224, 224))
    # Convert the image to a numpy array
    image = np.array(image)
    # Reshape the image to the expected input shape
    image = image.reshape((1, 224, 224, 3))
    # Make the prediction
    predictions = model.predict(image)
    # Convert the predictions to class labels
    class_labels = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
                    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
                    "Consolidation", "No Finding"]
    predicted_class_labels = [class_labels[i] for i in np.argmax(predictions, axis=1)[0]]
    return predicted_class_labels

# Create the Streamlit app
st.title("X-Ray Classification App")
st.write("Upload an X-ray image to make a prediction:")

# Create a file uploader
image_file = st.file_uploader("Select an image", type=["jpg", "jpeg", "png"])

# Create a button to trigger the prediction
if st.button("Make Prediction"):
    if image_file is not None:
        # Load the uploaded image
        image = Image.open(image_file)
        # Convert the image to a numpy array
        image = np.array(image)
        # Resize the image to the expected size
        image = tf.image.resize(img, (320, 320))
        # Convert the image to a numpy array
        image = np.array(image)
        # Reshape the image to the expected input shape
        image = image.reshape((1, 320, 320, 3))
        # Make the prediction
        predictions = model.predict(image)
        # Convert the predictions to class labels
        class_labels = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
                          "Mass", "Nodule", "Pneumonia", "Pneumothorax",
                          "Consolidation", "No Finding"]
        predicted_class_labels = [class_labels[i] for i in np.argmax(predictions, axis=1)[0]]
        # Display the predicted class labels
        st.write("Predicted class labels:", predicted_class_labels)
    else:
        st.write("Please upload an image.")