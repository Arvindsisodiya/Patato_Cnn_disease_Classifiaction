from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, UnidentifiedImageError

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "potato_cnn_model.keras"
CLASS_NAMES_PATH = APP_DIR / "classification_report.txt"
IMAGE_SIZE = (128, 128)  # Matches IMG_SIZE in cnn_code_13.ipynb

st.set_page_config(page_title="Potato Leaf Disease Classifier", page_icon="🌱", layout="centered")

st.markdown("""
<style>

/* =================================
   MAIN BACKGROUND
================================= */
.stApp {
    background-color: #C8EDC7;
}

/* MAIN CONTENT */
.block-container {
    max-width: 1000px;
    padding-top: 3rem;
    padding-bottom: 3rem;
}
/* UPLOAD TEXT */
[data-testid="stFileUploader"] label {
    color: white !important;
    font-size: 18px !important;
    font-weight: 700 !important;
}

[data-testid="stFileUploaderDropzone"] span {
    color: #1C1C4F !important;
}
/* =================================
   HEADINGS
================================= */
h1, h2, h3, h4 {
    color: #1C1C4F !important;
    font-weight: 800 !important;
}

h1 {
    text-align: center;
    font-size: 42px !important;
}

/* NORMAL TEXT */
.stMarkdown p,
.stMarkdown span,
.stCaption,
[data-testid="stCaptionContainer"] {
    color: #1C1C4F !important;
}

/* =================================
   FILE UPLOADER
================================= */
[data-testid="stFileUploader"] {
    background-color: #7E7EE6 !important;
    border: 2px dashed #1C1C4F;
    border-radius: 15px;
    padding: 20px;
}

/* UPLOAD DROPZONE - REMOVE BLACK AREA */
[data-testid="stFileUploaderDropzone"] {
    background-color: #C8EDC7 !important;
    border: 2px dashed #1C1C4F !important;
    border-radius: 12px;
}

/* UPLOADER TEXT */
[data-testid="stFileUploaderDropzone"] span,
[data-testid="stFileUploaderDropzone"] small {
    color: #1C1C4F !important;
}

/* UPLOADER BUTTON */
[data-testid="stFileUploader"] button {
    background-color: #1C1C4F !important;
    color: white !important;
    border-radius: 8px;
}

/* =================================
   PREDICTION BUTTON
================================= */
.stButton > button {
    background-color: #7E7EE6 !important;
    color: white !important;
    border: 2px solid #1C1C4F;
    border-radius: 12px;
    width: 100%;
    height: 50px;
    font-size: 17px;
    font-weight: 700;
}

/* BUTTON HOVER */
.stButton > button:hover {
    background-color: #1C1C4F !important;
    color: white !important;
    border: 2px solid #1C1C4F;
}

/* =================================
   PREDICTION RESULT 
================================= */
[data-testid="stAlert"] {
    background-color: #7E7EE6 !important;
    border: 2px solid #1C1C4F !important;
    border-radius: 15px;
}

[data-testid="stAlert"] p,
[data-testid="stAlert"] span,
[data-testid="stAlert"] strong {
    color: #FFFFFF !important;
    font-weight: 700 !important;
}

/* =================================
   MODEL SCORE CARD
================================= */
[data-testid="stMetric"] {
    background-color: #7E7EE6 !important;
    border: 2px solid #1C1C4F;
    border-radius: 15px;
    padding: 18px;
}

/* METRIC LABEL */
[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] p {
    color: white !important;
}

/* METRIC VALUE */
[data-testid="stMetricValue"],
[data-testid="stMetricValue"] div {
    color: white !important;
    font-weight: 800 !important;
}

/* =================================
   PROGRESS LABELS - DARK NAVY
================================= */
[data-testid="stProgress"] p,
[data-testid="stProgress"] label,
[data-testid="stProgress"] span {
    color: #1C1C4F !important;
    font-weight: 700 !important;
}

/* PROGRESS BAR TRACK */
[data-testid="stProgressBar"] {
    background-color: #C8EDC7 !important;
}

/* PROGRESS BAR FILL */
[data-testid="stProgressBar"] > div > div {
    background-color: #7E7EE6 !important;
}

/* =================================
   LOADING SPINNER
================================= */
[data-testid="stSpinner"] {
    background-color: #7E7EE6 !important;
    border: 2px solid #1C1C4F;
    border-radius: 12px;
    padding: 15px;
}

[data-testid="stSpinner"] p {
    color: #1C1C4F !important;
    font-weight: 700 !important;
}

/* SPINNER ICON */
[data-testid="stSpinner"] svg {
    color: #1C1C4F !important;
}

/* =================================
   CAPTION
================================= */
.stCaption {
    color: #1C1C4F !important;
    text-align: center;
    font-weight: 600;
}

/* =================================
   IMAGE
================================= */
[data-testid="stImage"] {
    border-radius: 15px;
}

/* =================================
   DIVIDER
================================= */
hr {
    border-color: #7E7EE6 !important;
}

</style>
""", unsafe_allow_html=True)

st.title("🌿 Potato Leaf Disease Classifier")
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


# ==========================================
# IMAGE UPLOAD
# ==========================================

upload = st.file_uploader(
    "Choose a leaf image",
    type=["jpg", "jpeg", "png"]
)

if upload is not None:

    try:
        image = Image.open(upload).convert("RGB")

    except (UnidentifiedImageError, OSError):
        st.error("Invalid image! Please upload a JPG or PNG.")
        st.stop()

    # Display uploaded image
    st.image(
        image,
        caption="Uploaded leaf",
        width="stretch"
    )

    # ==========================================
    # PREDICTION BUTTON
    # ==========================================

    if st.button("🔍 Predict disease", type="primary"):

        # Show loading spinner
        with st.spinner("🌱 Processing image... Please wait"):

            # Resize image
            resized = image.resize(IMAGE_SIZE)

            # Prepare image for CNN
            batch = np.expand_dims(
                np.asarray(resized, dtype=np.float32),
                axis=0
            )

            # Get model predictions
            scores = np.asarray(
                model.predict(batch, verbose=0)
            )[0].reshape(-1)

        # ==========================================
        # CHECK MODEL OUTPUT
        # ==========================================

        if len(scores) != len(class_names):

            st.error(
                f"Model returned {len(scores)} scores, "
                f"but there are {len(class_names)} class labels."
            )

            st.stop()

        # ==========================================
        # GET BEST PREDICTION
        # ==========================================

        best_index = int(np.argmax(scores))

        predicted_class = display_name(
            class_names[best_index]
        )

        predicted_score = float(scores[best_index])

        # ==========================================
        # DISPLAY PREDICTION RESULT
        # ==========================================

        st.success(
            f"Prediction: **{predicted_class}**"
        )

        # ==========================================
        # DISPLAY MODEL SCORE
        # ==========================================

        st.metric(
            "Model score",
            f"{predicted_score:.1%}"
        )

        # ==========================================
        # DISPLAY ALL CLASS SCORES
        # ==========================================

        st.subheader(
            "Scores of classes based on the model's output"
        )

        for label, score in zip(class_names, scores):

            st.progress(
                float(np.clip(score, 0.0, 1.0)),
                text=f"{display_name(label)} — {score:.1%}"
            )

# ==========================================
# FOOTER
# ==========================================

st.caption(
    "🌱 AI-assisted prediction — For informational purposes only."
)
