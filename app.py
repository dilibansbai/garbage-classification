import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

model = load_model("models/best_model.h5")

class_labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

st.title("♻️ Garbage Classification App")

st.write("Upload an image to classify waste type")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)[0]

    predicted_class = class_labels[np.argmax(predictions)]
    confidence = np.max(predictions)

    st.subheader("Prediction")
    st.write(f"Category: **{predicted_class}**")
    st.write(f"Confidence: **{confidence:.2f}**")

    st.subheader("Top 3 Predictions")

    top_3_idx = np.argsort(predictions)[-3:][::-1]

    for i in top_3_idx:
        st.write(f"{class_labels[i]}: {predictions[i]:.2f}")