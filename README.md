# 😷 Face Mask Detector — Complete Project

## 📁 Project Structure
```
face_mask_detector/
├── dataset/
│   ├── with_mask/       ← ~3800 images from Kaggle
│   └── without_mask/    ← ~3800 images from Kaggle
├── train_model.py       ← STEP 1: Train the model
├── detect_mask.py       ← STEP 2: Webcam real-time detection
├── app.py               ← STEP 3: Streamlit web app
├── mask_detector.model  ← Auto-generated after training
└── training_plot.png    ← Auto-generated after training
```

---

## 🎓 What Topics Each File Teaches You

### train_model.py
| Code Section | Topic to Learn |
|---|---|
| `import numpy, matplotlib` | Python Libraries |
| `train_test_split` | ML Concepts: Overfitting, Train-Test Split |
| `LabelBinarizer` | Label Encoding |
| `ImageDataGenerator` | Data Augmentation |
| `MobileNetV2(weights="imagenet")` | Transfer Learning, CNNs |
| `layer.trainable = False` | Freezing Layers |
| `Dense, Dropout, Flatten` | Neural Network Architecture |
| `model.compile(optimizer, loss)` | Gradient Descent, Loss Functions |
| `model.fit(...)` | Model Training, Epochs, Batch Size |
| `model.save(...)` | Saving/Loading Models |

### detect_mask.py
| Code Section | Topic to Learn |
|---|---|
| `cv2.VideoCapture(0)` | OpenCV: Webcam Access |
| `cv2.cvtColor(...)` | Color Spaces (BGR, RGB, Gray) |
| `face_detector.detectMultiScale(...)` | Face Detection, Haar Cascades |
| `np.expand_dims(...)` | NumPy Array Shapes |
| `model.predict(...)` | Model Inference |
| `cv2.rectangle, cv2.putText` | Drawing on Images |

### app.py
| Code Section | Topic to Learn |
|---|---|
| `st.file_uploader` | File I/O with Streamlit |
| `@st.cache_resource` | Caching, App Performance |
| `Image.open(...)` | PIL/Pillow Library |
| `np.array(pil_image)` | Image → NumPy Conversion |
| Full pipeline | Model Deployment |

---

## 🚀 How to Run

### Step 1 — Download Dataset
1. Go to: https://www.kaggle.com/datasets/omkargurav/face-mask-dataset
2. Download and unzip into `dataset/` folder
3. Make sure you have `dataset/with_mask/` and `dataset/without_mask/`

### Step 2 — Install Dependencies
```bash
pip install tensorflow keras opencv-python numpy matplotlib streamlit pillow scikit-learn
```

### Step 3 — Train the Model
```bash
python train_model.py
```
*(Takes 5–15 minutes depending on your machine)*

### Step 4 — Run Webcam Detection
```bash
python detect_mask.py
```
*Press Q to quit*

### Step 5 — Launch Web App
```bash
streamlit run app.py
```
*Opens in browser at http://localhost:8501*

---

## 📚 Learning Roadmap (Do in This Order)

### Week 1 — Python Foundations
- [ ] Lists, loops, functions, dictionaries
- [ ] NumPy basics: arrays, shapes, indexing
- [ ] Matplotlib: plotting graphs

### Week 2 — OpenCV
- [ ] Reading images and video
- [ ] Color spaces (BGR vs RGB vs Grayscale)
- [ ] Drawing shapes and text on images
- [ ] Face detection with Haar Cascades

### Week 3 — ML & Deep Learning Concepts
- [ ] What is Machine Learning?
- [ ] Classification vs Regression
- [ ] Train/Test split, overfitting
- [ ] What is a Neural Network? (Watch 3Blue1Brown series)
- [ ] What is a CNN? Why for images?
- [ ] Transfer Learning concept

### Week 4 — Keras / TensorFlow
- [ ] Building a Sequential model
- [ ] Dense, Dropout, Flatten layers
- [ ] model.compile() — loss functions, optimizers
- [ ] model.fit() — training
- [ ] model.predict() — inference
- [ ] model.save() / load_model()

### Week 5 — Project Build
- [ ] Train the model (train_model.py)
- [ ] Test on webcam (detect_mask.py)
- [ ] Deploy web app (app.py)

### Week 6 — Polish for Resume
- [ ] Push to GitHub with clean README
- [ ] Deploy on Streamlit Cloud (free)
- [ ] Note accuracy achieved in resume bullet point

---

## 📝 How to Write This on Your Resume

```
Face Mask Detection System                          [GitHub Link] [Live Demo Link]
- Built an end-to-end real-time face mask detector using MobileNetV2 (Transfer Learning)
- Trained on 7,500+ images achieving ~98% accuracy on test set
- Integrated OpenCV for webcam-based real-time detection
- Deployed as an interactive web application using Streamlit
Tech Stack: Python, TensorFlow/Keras, OpenCV, NumPy, Streamlit
```

---

## 🔗 Free Resources
- **NumPy**: https://numpy.org/learn/
- **OpenCV**: https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html
- **3Blue1Brown Neural Networks**: https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi
- **Keras Docs**: https://keras.io/guides/
- **Streamlit Docs**: https://docs.streamlit.io/
- **Dataset**: https://www.kaggle.com/datasets/omkargurav/face-mask-dataset
