# ============================================================
# detect_mask.py — Real-time Face Mask Detection via Webcam
# TOPICS: OpenCV, Face Detection, Model Inference,
#         Real-time Video Processing, NumPy
# ============================================================

# --- TOPIC: Importing Libraries ---
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# -------------------------------------------------------
# TOPIC: Loading Pre-trained Models
# 1. OpenCV's face detector (finds faces in frames)
# 2. Our trained mask detector model
# -------------------------------------------------------
print("[INFO] Loading face detector model...")

# OpenCV provides a pre-trained face detector using deep learning
# prototxt = model architecture, caffemodel = trained weights
FACE_PROTO = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_detector = cv2.CascadeClassifier(FACE_PROTO)

print("[INFO] Loading mask detector model...")
mask_model = load_model("mask_detector.h5")
# -------------------------------------------------------
# TOPIC: Starting Webcam Feed
# cv2.VideoCapture(0) opens your default webcam
# -------------------------------------------------------
print("[INFO] Starting webcam...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[ERROR] Cannot open webcam. Check if it's connected.")
    exit()

print("[INFO] Press 'q' to quit.")

# -------------------------------------------------------
# TOPIC: Real-time Video Loop
# Every frame from the webcam is:
# 1. Read → 2. Detect Faces → 3. Predict Mask → 4. Display
# -------------------------------------------------------
while True:
    # Read one frame from webcam
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to grab frame.")
        break

    # TOPIC: Color Space Conversion
    # OpenCV reads in BGR, but many operations need RGB or Gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # -------------------------------------------------------
    # TOPIC: Face Detection with OpenCV
    # detectMultiScale finds all faces in the frame
    # Returns list of (x, y, w, h) bounding boxes
    # -------------------------------------------------------
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,    # How much image is reduced at each scale
        minNeighbors=5,     # How many neighbors each rectangle should have
        minSize=(60, 60)    # Minimum face size to detect
    )

    # -------------------------------------------------------
    # TOPIC: Processing Each Detected Face
    # For every face found, we crop it and pass to mask model
    # -------------------------------------------------------
    for (x, y, w, h) in faces:
        # Crop face from frame
        face_roi = rgb_frame[y:y+h, x:x+w]

        # Preprocess: resize to 224x224, normalize for MobileNetV2
        face_roi = cv2.resize(face_roi, (224, 224))
        face_roi = img_to_array(face_roi)
        face_roi = preprocess_input(face_roi)
        face_roi = np.expand_dims(face_roi, axis=0)  # Add batch dimension

        # -------------------------------------------------------
        # TOPIC: Model Inference (Prediction)
        # model.predict returns a probability between 0 and 1
        # -------------------------------------------------------
        prediction = mask_model.predict(face_roi)[0][0]

        # Map prediction to label
        # (depends on how LabelBinarizer encoded in training)
        if prediction > 0.5:
            label = "No Mask"
            color = (0, 0, 255)    # Red in BGR
        else:
            label = "Mask"
            color = (0, 255, 0)    # Green in BGR

        confidence = prediction if prediction > 0.5 else 1 - prediction
        label_text = f"{label}: {confidence*100:.1f}%"

        # -------------------------------------------------------
        # TOPIC: Drawing on Frames with OpenCV
        # Draw bounding box around face + label text
        # -------------------------------------------------------
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(
            frame, label_text,
            (x, y - 10),                    # Position above box
            cv2.FONT_HERSHEY_SIMPLEX,       # Font style
            0.7,                            # Font size
            color,                          # Font color
            2                               # Thickness
        )

    # -------------------------------------------------------
    # TOPIC: Displaying the Frame
    # imshow renders the frame in a window
    # -------------------------------------------------------
    cv2.imshow("Face Mask Detector — Press Q to Quit", frame)

    # Break the loop if user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -------------------------------------------------------
# TOPIC: Cleanup
# Always release webcam and close windows
# -------------------------------------------------------
print("[INFO] Shutting down...")
cap.release()
cv2.destroyAllWindows()
