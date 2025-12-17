import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import gdown
import os

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Alzheimer’s Disease Detection",
    page_icon="🧠",
    layout="centered"
)

# ================= COLORFUL CSS =================
st.markdown("""
<style>
/* Background */
.stApp {
    background: linear-gradient(135deg, #e0f2ff, #fef6ff);
}

/* Title */
.title-text {
    text-align: center;
    font-size: 42px;
    font-weight: 800;
    background: linear-gradient(90deg, #1c92d2, #f2fcfe);
    -webkit-background-clip: text;
    color: transparent;
}

/* Subtitle */
.subtitle {
    text-align: center;
    font-size: 18px;
    color: #4a4a4a;
    margin-bottom: 20px;
}

/* Card */
.card {
    background: white;
    padding: 25px;
    border-radius: 18px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.12);
    margin-bottom: 20px;
}

/* Result Box */
.result-box {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    padding: 25px;
    border-radius: 16px;
    text-align: center;
    box-shadow: 0 10px 25px rgba(118,75,162,0.4);
}

/* Upload Box */
.upload-box {
    border: 2px dashed #6a5acd;
    padding: 20px;
    border-radius: 14px;
    background: #f9f7ff;
}

/* Footer */
.footer {
    text-align: center;
    font-size: 13px;
    color: #666;
    margin-top: 30px;
}

/* Progress label */
.progress-label {
    font-weight: 600;
    color: #333;
}
</style>
""", unsafe_allow_html=True)

# ================= GOOGLE DRIVE MODEL =================
MODEL_URL = "1MxkGejVi-1LmT8Q9r26laxzYiQRMdGUj"
MODEL_PATH = "AugmentedAlzheimer.h5"

@st.cache_resource
def load_alzheimer_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("📥 Downloading AI model..."):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    return load_model(MODEL_PATH)

model = load_alzheimer_model()

# ================= CLASS NAMES =================
class_names = [
    "Mild Demented",
    "Moderate Demented",
    "Non Demented",
    "Very Mild Demented"
]

# ================= HEADER =================
st.markdown("<div class='title-text'>🧠 Alzheimer’s Disease Detection</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>AI-powered MRI Image Classification</div>", unsafe_allow_html=True)

# ================= INPUT CARD =================
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("### 📥 Upload MRI Image")

uploaded_image = st.file_uploader(
    "Choose an MRI image",
    type=["jpg", "jpeg", "png"]
)

camera_image = st.camera_input("OR Capture using Camera")

st.markdown("</div>", unsafe_allow_html=True)

# ================= IMAGE HANDLING =================
img = None
if uploaded_image:
    img = Image.open(uploaded_image).convert("RGB")
elif camera_image:
    img = Image.open(camera_image).convert("RGB")

# ================= PREDICTION =================
if img:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.image(img, caption="🖼️ Input MRI Image", use_container_width=True)

    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)[0]
    index = int(np.argmax(prediction))

    result = class_names[index]
    confidence = prediction[index] * 100

    st.markdown("</div>", unsafe_allow_html=True)

    # ================= RESULT =================
    st.markdown("<div class='result-box'>", unsafe_allow_html=True)
    st.markdown(f"## 🔬 Prediction: **{result}**")
    st.markdown(f"### 🎯 Confidence: **{confidence:.2f}%**")
    st.markdown("</div>", unsafe_allow_html=True)

    # ================= PROBABILITY BARS =================
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 📊 Class Probabilities")

    for i, cls in enumerate(class_names):
        st.markdown(f"<div class='progress-label'>{cls}</div>", unsafe_allow_html=True)
        st.progress(int(prediction[i] * 100))

    st.markdown("</div>", unsafe_allow_html=True)

# ================= FOOTER =================
st.markdown(
    "<div class='footer'>⚕️ For educational use only — Not a medical diagnosis</div>",
    unsafe_allow_html=True
)




