import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
import tensorflow as tf
from PIL import Image

# Load the model
model = load_model('brain_tumor_resnet50v2_model.keras')

# Class names based on your dataset
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# App title
st.title("Brain Tumor Classification")

# Upload image
uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load and process image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', width=700)  # Use width instead of use_container_width
    st.write("")
    st.write("Classifying...")

    # Resize and convert to array
    image = image.resize((150, 150))  # Resize to match model input
    img_array = img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize

    # Make prediction
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    predicted_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)

    # Show results
    st.write(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}%")
else:
    st.write("Please upload an image to classify.")