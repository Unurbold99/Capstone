import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import joblib
import torch
from fastai.vision.all import *
from fastcore.all import *
import pickle


learn_inf = load_learner("export.pkl")

#Prediction function
def classify_img(data):
    pred, pred_idx, probs = learn_inf.predict(data)
    return pred, probs[pred_idx]

#Text on screen
st.title("X-Ray Classifier")
st.write("Upload an X-ray image to make a prediction:")

#Turning the picture into Bytes_data was the biggest hurdle that I encountered in this Project.
bytes_data = None
uploaded_image = st.file_uploader("Upload Picture:")
if uploaded_image:
    bytes_data = uploaded_image.getvalue()
    st.image(bytes_data, caption="Uploaded image")   
if bytes_data:
    classify = st.button("CLASSIFY!")
    if classify:
        label, confidence = classify_img(bytes_data)
        st.write(f"Classification: {label}! Accuracy: ({confidence:.04f})")
        st.write("NOTE: This prediction should not be taken as a medical advise and is only for scholary purposes.")


# Overall this Streamlit App was quite hard to make it work, at the end it ended up working at the least.
