from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, UnidentifiedImageError

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "potato_cnn_model.keras"
CLASS_NAMES_PATH = APP_DIR / "classification_report.txt"
IMAGE_SIZE = (128, 128)  # Matches IMG_SIZE in cnn_code_13.ipynb

st.set_page_config(page_title="Potato Leaf Classifier", page_icon="🥔", layout="centered")
st.title("🥔 Potato Leaf Disease Classifier")
st.write("Upload a potato leaf photo to see the CNN prediction and its output score.")


@st.cache_resource
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    return tf.keras.models.load_model(MODEL_PATH)


def load_class_names():
    if not CLASS_NAMES_PATH.is_file():
        raise FileNotFoundError(f"Class labels not found: {CLASS_NAMES_PATH}")
    names = [line.strip() for line in CLASS_NAMES_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not names:
        raise ValueError("classification_report.txt is empty")
    return names


def display_name(label: str) -> str:
    return label.replace("Potato___", "").replace("_", " ").title()


try:
    model = load_model()
    class_names = load_class_names()
except Exception as exc:
    st.error(f"Could not load the model or class labels: {exc}")
    st.info("Keep app.py, potato_cnn_model.keras, and classification_report.txt together in this folder.")
    st.stop()

upload = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png"])
if upload is not None:
    try:
        image = Image.open(upload).convert("RGB")
    except (UnidentifiedImageError, OSError):
        st.error("That file could not be opened as an image. Please choose a JPG or PNG.")
        st.stop()

    st.image(image, caption="Uploaded leaf", use_container_width=True)
    if st.button("Predict disease", type="primary"):
        resized = image.resize(IMAGE_SIZE)
        # Keep raw 0–255 pixel values: the trained model has Rescaling(1/255) inside it.
        batch = np.expand_dims(np.asarray(resized, dtype=np.float32), axis=0)
        scores = np.asarray(model.predict(batch, verbose=0))[0].reshape(-1)

        if len(scores) != len(class_names):
            st.error(f"Model returned {len(scores)} scores, but there are {len(class_names)} class labels.")
            st.stop()

        best_index = int(np.argmax(scores))
        st.success(f"Prediction: **{display_name(class_names[best_index])}**")
        st.metric("Model score", f"{float(scores[best_index]):.1%}")
        st.subheader("Scores of classes basesd on the model's output")
        for label, score in zip(class_names, scores):
            st.progress(float(np.clip(score, 0.0, 1.0)), text=f"{display_name(label)} — {score:.1%}")

st.caption("The score is the model's output, not a confirmed diagnosis. Ask an agricultural expert if you are unsure.")
