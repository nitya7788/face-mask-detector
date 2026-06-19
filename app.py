# ============================================================
# app.py — Streamlit Web App for Face Mask Detection
# TOPICS: Streamlit, File Upload, PIL/Pillow, Model Deployment
# ============================================================

# --- TOPIC: Importing Libraries ---
import streamlit as st
import numpy as np
import cv2
from PIL import Image                          # Pillow: image handling in Python
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# -------------------------------------------------------
# TOPIC: Streamlit Page Setup
# st.set_page_config sets browser tab title and layout
# -------------------------------------------------------
st.set_page_config(
    page_title="Face Mask Detector",
    page_icon="😷",
    layout="centered"
)

st.title("😷 Face Mask Detector")
st.markdown("Upload a photo to detect whether the person is wearing a mask.")
st.markdown("---")

# -------------------------------------------------------
# TOPIC: Caching — Load Model Once
# @st.cache_resource loads the model only once
# Without this, it reloads every time the app reruns (slow!)
# -------------------------------------------------------
@st.cache_resource
def load_mask_model():
    return load_model("mask_detector.h5")

@st.cache_resource
def load_face_detector():
    proto = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    return cv2.CascadeClassifier(proto)

model = load_mask_model()
face_detector = load_face_detector()

# -------------------------------------------------------
# TOPIC: File Upload with Streamlit
# st.file_uploader lets users upload images through the browser
# -------------------------------------------------------
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)

# -------------------------------------------------------
# TOPIC: Image Processing Pipeline
# Only runs when user uploads a file
# -------------------------------------------------------
if uploaded_file is not None:
    # TOPIC: PIL / Pillow — Open image from uploaded file
    pil_image = Image.open(uploaded_file).convert("RGB")
    st.image(pil_image, caption="Uploaded Image", use_column_width=True)

    # Convert PIL Image → NumPy array → OpenCV format (BGR)
    image_np = np.array(pil_image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # -------------------------------------------------------
    # TOPIC: Face Detection
    # Same as in detect_mask.py — detect all faces in image
    # -------------------------------------------------------
    faces = face_detector.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )

    if len(faces) == 0:
        st.warning("⚠️ No face detected. Please upload a clearer front-facing photo.")
    else:
        st.success(f"✅ Detected {len(faces)} face(s). Analyzing...")

        result_image = image_np.copy()

        for (x, y, w, h) in faces:
            # Crop and preprocess face
            face_roi = image_np[y:y+h, x:x+w]
            face_roi_resized = cv2.resize(face_roi, (224, 224))
            face_array = img_to_array(face_roi_resized)
            face_array = preprocess_input(face_array)
            face_array = np.expand_dims(face_array, axis=0)

            # -------------------------------------------------------
            # TOPIC: Prediction + Confidence Score
            # -------------------------------------------------------
            prediction = model.predict(face_array)[0][0]

            if prediction > 0.5:
                label = "No Mask ❌"
                confidence = prediction
                color = (255, 0, 0)    # Red in RGB
            else:
                label = "Mask ✅"
                confidence = 1 - prediction
                color = (0, 200, 0)    # Green in RGB

            label_text = f"{label}: {confidence*100:.1f}%"

            # Draw rectangle + label on result image
            cv2.rectangle(result_image, (x, y), (x+w, y+h), color, 3)
            cv2.putText(
                result_image, label_text,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, color, 2
            )

        # -------------------------------------------------------
        # TOPIC: Displaying Result in Streamlit
        # -------------------------------------------------------
        st.image(result_image, caption="Detection Result", use_column_width=True)

        # Summary card
        st.markdown("### 🔍 Result Summary")
        for i, (x, y, w, h) in enumerate(faces):
            face_roi = image_np[y:y+h, x:x+w]
            face_roi_resized = cv2.resize(face_roi, (224, 224))
            face_array = img_to_array(face_roi_resized)
            face_array = preprocess_input(face_array)
            face_array = np.expand_dims(face_array, axis=0)
            prediction = model.predict(face_array)[0][0]

            if prediction > 0.5:
                st.error(f"**Face {i+1}:** No Mask Detected — {prediction*100:.1f}% confidence")
            else:
                st.success(f"**Face {i+1}:** Mask Detected — {(1-prediction)*100:.1f}% confidence")

# -------------------------------------------------------
# TOPIC: Sidebar Info
# -------------------------------------------------------
st.sidebar.header("ℹ️ About this App")
st.sidebar.markdown("""
This app uses:
- **MobileNetV2** (transfer learning)
- **OpenCV** for face detection
- **Streamlit** for the web interface

Built as a resume project to demonstrate:
- Deep Learning
- Computer Vision
- Model Deployment
""")
