import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import decode_predictions
from fastai.vision.all import *

# Load the model
model = tf.keras.models.load_model('model.h5')

# Define a function to make predictions
def make_prediction(image):
    # Load the image
    img = load_img(image, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    predictions = decode_predictions(predictions, top=3)[0]
    return predictions

# Define the class labels
class_labels = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
                 "Mass", "Nodule", "Pneumonia", "Pneumothorax",
                 "Consolidation", "No Finding"]

# Create the Streamlit app
st.title("Image Classifier")
st.write("Upload an image to classify it")

# Create a file uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# Create a button to make the prediction
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    predictions = make_prediction(uploaded_file)
    st.write("Predictions:")
    for prediction in predictions:
        st.write(f"{class_labels[int(prediction[1]) - 1]}: {prediction[2]}")