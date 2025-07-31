import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os

# Load TFLite model once and cache it
@st.cache_resource
def load_tflite_model():
    interpreter = tf.lite.Interpreter(model_path="model/fish_classifier.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Class names
class_names = ['Bangus', 'Big Head Carp', 'Black Spotted Barb', 'Catfish', 'Climbing Perch', 
               'Fourfinger Threadfin', 'Freshwater Eel', 'Glass Perchlet', 'Goby', 'Gold Fish', 
               'Gourami', 'Grass Carp', 'Green Spotted Puffer', 'Indian Carp', 'Indo-Pacific Tarpon',
               'Jaguar Gapote', 'Janitor Fish', 'Knifefish', 'Long-Snouted Pipefish', 'Mosquito Fish',
               'Mudfish', 'Mullet', 'Pangasius', 'Perch', 'Scat Fish', 'Silver Barb', 'Silver Carp',
               'Silver Perch', 'Snakehead', 'Tenpounder', 'Tilapia']

# Preprocess function
def prepare_image(uploaded_image):
    img = uploaded_image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
    return img_array

# UI
st.set_page_config(page_title="Fish Classifier", page_icon="🐟")
st.title("🐟 Fish Classifier")
st.write("Upload an image of a fish to classify its species.")

uploaded_file = st.file_uploader("Choose a fish image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file)
    img_array = prepare_image(img)

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    prediction_scores = output_data[0]
    max_score = np.max(prediction_scores)
    predicted_index = np.argmax(prediction_scores)
    confidence = round(float(max_score) * 100, 2)

    if max_score >= 0.5:
        predicted_class = class_names[predicted_index]
        st.success(f"Prediction: {predicted_class}")
        st.info(f"Confidence: {confidence}%")
        st.image(img, caption="Uploaded Image", use_container_width=True)
    else:
        st.error("No fish detected (confidence below 50%)")