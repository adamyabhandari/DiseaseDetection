import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image

# Load the trained models
MODEL_PATHS = {
    "Retinal Disease Detection": "retinal_disease_detection_model.h5",
    "Pneumonia Detection": "chest_disease_detection_model.h5",
    "tb_disease Detection": "tb_disease_detection_model.h5"
}

# Load all models into a dictionary
models = {name: tf.keras.models.load_model(path) for name, path in MODEL_PATHS.items()}

# Set image dimensions
IMG_SIZE = 128  # Update to match the model's expected input size

# Function to preprocess the image
def preprocess_image(image):
    image = np.array(image)  # Convert PIL image to numpy array

    # Convert grayscale images to RGB
    if len(image.shape) == 2:  # Grayscale image (height, width)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:  # RGBA image (height, width, 4)
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    elif image.shape[2] == 1:  # Single-channel image (height, width, 1)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))  # Resize to match model input size
    image = image / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to make prediction
def predict_disease(image, model):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)

    # Extract the predicted probability
    predicted_probability = prediction[0][0]  # Assuming binary classification

    # Compare the probability to the threshold
    return "❌ Disease Detected" if predicted_probability > 0.5 else "✅ No Disease Detected"

# Streamlit UI
st.title("🩺 AI-Powered Early Disease Detection")
st.write("Upload a medical image (X-ray, MRI, Retina scan, etc.) to detect early-stage diseases.")

# Dropdown to select the model
selected_model_name = st.selectbox("Select a Disease Model:", list(MODEL_PATHS.keys()))
selected_model = models[selected_model_name]

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Run detection when the button is clicked
    if st.button("Analyze Image"):
        result = predict_disease(image, selected_model)
        st.subheader("🔍 Prediction Result")
        st.write(f"**{selected_model_name}**: {result}")
