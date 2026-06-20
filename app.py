import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Page config
st.set_page_config(page_title="Potato Disease Classifier", page_icon="🥔")

# Load model (cached so it doesn't reload every time)
@st.cache_resource
def load_my_model():
    return load_model("potato_disease_model.h5")

model = load_my_model()

IMG_SIZE = 128
class_labels = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

# UI
st.title("Potato Leaf Disease Classifier")
st.write("Upload a potato leaf image to detect disease.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img_resized = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.subheader("Result")
    st.write(f"**Prediction:** {predicted_class.replace('Potato___', '').replace('_', ' ')}")
    st.write(f"**Confidence:** {confidence:.2f}%")