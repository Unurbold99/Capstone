import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import joblib
import torch
from fastai.vision.all import *
from fastcore.all import *
import pickle

## LOAD MODEL
learn_inf = load_learner("export.pkl")

## CLASSIFIER
def classify_img(data):
    pred, pred_idx, probs = learn_inf.predict(data)
    return pred, probs[pred_idx]

## STREAMLIT
st.title("X-Ray Classifier")

# Set up layout
col1, col2, col3 = st.columns([1, 4, 1])

with col1:
    st.write("")

with col2:
    uploaded_image = st.file_uploader("Upload Picture:", accept_multiple_files=False)
    if uploaded_image:
        bytes_data = uploaded_image.getvalue()
        st.image(bytes_data, caption="Uploaded image")

with col3:
    st.write("")

if uploaded_image:
    st.write("")

# Center align the button
col1, col2, col3 = st.columns([1, 4, 1])
with col1:
    st.write("")

with col2:
    classify = st.button("CLASSIFY!", key="classify_button", help="Click to classify the uploaded image",)
    st.write("")
    st.write("")

with col3:
    st.write("")

if classify:
    label, confidence = classify_img(bytes_data)
    st.write("")

    # Center align the result and increase font size
    st.markdown(f"<p style='text-align: center; font-size: 24px;'>It is a {label}! (Accuracy: {confidence:.04f})</p>", unsafe_allow_html=True)


        
