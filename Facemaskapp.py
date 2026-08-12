import os

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_mask_detector_model.h5")


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


def predict_mask(model, image: Image.Image):
    rgb_image = image.convert("RGB")
    image_array = np.array(rgb_image)
    resized_image = cv2.resize(image_array, (128, 128))
    normalized_image = resized_image / 255.0
    reshaped_image = np.expand_dims(normalized_image, axis=0)

    prediction = model.predict(reshaped_image, verbose=0)[0]
    predicted_index = int(np.argmax(prediction))
    confidence = float(prediction[predicted_index])

    labels = ["No Mask", "Mask"]
    label = labels[predicted_index]
    return label, confidence, prediction


st.set_page_config(page_title="Face Mask Detector", page_icon="😷", layout="centered")

st.title("Face Mask Detection")
st.caption("Upload a face image to detect whether the person is wearing a mask or not.")

uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    model = load_model()
    label, confidence, probabilities = predict_mask(model, image)

    result_text = f"Result: {label}"
    if label == "Mask":
        status_color = "#22c55e"
    else:
        status_color = "#ef4444"

    st.markdown(
        f"<h3 style='color:{status_color};'>{result_text}</h3>",
        unsafe_allow_html=True,
    )
    st.progress(value=confidence, text=f"Confidence: {confidence * 100:.2f}%")
    st.write(f"Confidence: {confidence * 100:.2f}%")

    st.subheader("Prediction scores")
    for idx, score in enumerate(probabilities):
        label_name = ["No Mask", "Mask"][idx]
        st.write(f"{label_name}: {score * 100:.2f}%")

else:
    st.info("Please upload an image to begin detection.")

st.markdown("---")
st.write("Model file: face_mask_detector_model.h5")
