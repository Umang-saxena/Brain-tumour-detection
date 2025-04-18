import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
import tensorflow as tf

# Load the model
model = load_model('model.keras')

# Class names based on your dataset
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# App title
st.title("Brain Tumor Classification")

# Upload image
uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load and process image
    image = load_img(uploaded_file, target_size=(150, 150), color_mode='rgb')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Convert to array and normalize
    img_array = img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)
    img_array = img_array / 255.0

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