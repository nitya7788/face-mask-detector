# ============================================================
# train_model.py — Train the Face Mask Detector
# TOPICS: CNNs, Transfer Learning, MobileNetV2, Keras,
#         Data Augmentation, Train-Test Split, Model Saving
# ============================================================

# --- TOPIC: Importing Libraries ---
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
import os
import cv2

# Keras / TensorFlow imports
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# -------------------------------------------------------
# TOPIC: Hyperparameters
# These control how the model learns
# -------------------------------------------------------
INIT_LR = 1e-4
EPOCHS = 10
BATCH_SIZE = 8
     # Learning rate — how fast the model updates weights
          # How many times the model sees the full dataset
     # How many images are processed at once

# -------------------------------------------------------
# TOPIC: Loading & Preprocessing Data
# We read each image, resize it, and store label (mask/no mask)
# -------------------------------------------------------
print("[INFO] Loading images from dataset folder...")

DATASET_PATH = "dataset"   # Folder with 'with_mask' and 'without_mask' subfolders
CATEGORIES = ["with_mask", "without_mask"]

data = []    # Will store image pixel arrays
labels = []  # Will store corresponding labels

for category in CATEGORIES:
    folder_path = os.path.join(DATASET_PATH, category)
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        # Load image and resize to 224x224 (required by MobileNetV2)
        image = load_img(img_path, target_size=(224, 224))
        image = img_to_array(image)
        image = preprocess_input(image)  # Normalize pixels for MobileNetV2
        data.append(image)
        labels.append(category)

# Convert to NumPy arrays (ML models work with arrays, not Python lists)
data = np.array(data, dtype="float32")
labels = np.array(labels)

# -------------------------------------------------------
# TOPIC: Label Encoding
# Convert text labels ("with_mask") to numbers ([1, 0] or [0, 1])
# -------------------------------------------------------
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = np.array(labels, dtype="float32")  # shape: (N, 1) for binary

# -------------------------------------------------------
# TOPIC: Train-Test Split
# 80% for training, 20% for testing
# -------------------------------------------------------
(X_train, X_test, y_train, y_test) = train_test_split(
    data, labels, test_size=0.20, stratify=labels, random_state=42
)

# -------------------------------------------------------
# TOPIC: Data Augmentation
# Artificially expand training data by flipping, rotating, zooming images
# Helps the model generalize better
# -------------------------------------------------------
augmentation = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

# -------------------------------------------------------
# TOPIC: Transfer Learning with MobileNetV2
# MobileNetV2 is pre-trained on ImageNet (millions of images).
# We freeze its layers and only train our custom "head" on top.
# This saves time and gives better accuracy with less data.
# -------------------------------------------------------
print("[INFO] Building model with MobileNetV2 base...")

# Load MobileNetV2 WITHOUT its top classification layer
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,           # Remove the original classifier
    input_tensor=Input(shape=(224, 224, 3))
)

# Freeze the base model — we don't want to retrain it
for layer in base_model.layers:
    layer.trainable = False

# -------------------------------------------------------
# TOPIC: Custom Classification Head
# We add our own layers on top of MobileNetV2 for mask detection
# -------------------------------------------------------
head_model = base_model.output
head_model = AveragePooling2D(pool_size=(7, 7))(head_model)  # Reduce spatial size
head_model = Flatten(name="flatten")(head_model)              # 1D vector
head_model = Dense(128, activation="relu")(head_model)        # Fully connected layer
head_model = Dropout(0.5)(head_model)                         # Prevent overfitting
head_model = Dense(1, activation="sigmoid")(head_model)       # Output: 0 or 1

# Combine base + head into one model
model = Model(inputs=base_model.input, outputs=head_model)

# -------------------------------------------------------
# TOPIC: Compiling the Model
# Adam optimizer + binary_crossentropy loss for binary classification
# -------------------------------------------------------
print("[INFO] Compiling model...")
opt = Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# -------------------------------------------------------
# TOPIC: Training the Model
# -------------------------------------------------------
print("[INFO] Training model... (this may take a few minutes)")
history = model.fit(
    augmentation.flow(X_train, y_train, batch_size=BATCH_SIZE),
    steps_per_epoch=len(X_train) // BATCH_SIZE,
    validation_data=(X_test, y_test),
    validation_steps=len(X_test) // BATCH_SIZE,
    epochs=EPOCHS
)

# -------------------------------------------------------
# TOPIC: Evaluating the Model
# -------------------------------------------------------
print("[INFO] Evaluating model...")
pred = model.predict(X_test, batch_size=BATCH_SIZE)
pred = (pred > 0.5).astype("int32")  # Convert probabilities to 0 or 1

print(classification_report(y_test, pred, target_names=lb.classes_))

# -------------------------------------------------------
# TOPIC: Saving the Model
# Save trained model to disk so detect_mask.py can load it
# -------------------------------------------------------
print("[INFO] Saving model to mask_detector.model ...")
model.save("mask_detector.h5")
# -------------------------------------------------------
# TOPIC: Plotting Training Results
# -------------------------------------------------------
plt.style.use("ggplot")
plt.figure()
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Val Accuracy")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Loss / Accuracy")
plt.legend(loc="lower left")
plt.savefig("training_plot.png")
print("[INFO] Training plot saved to training_plot.png")
print("[DONE] Model trained and saved successfully!")
