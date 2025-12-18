# 🧠 Brain Tumor Detection Using CNN

An end-to-end **Medical Imaging AI system** that detects and classifies brain tumors from MRI scans using **Convolutional Neural Networks (CNNs)**.  
The project includes a **Flask-based web application** for real-time MRI upload and prediction, designed to assist healthcare professionals in early diagnosis.

---

## 🚀 Features
- MRI-based brain tumor classification
- Multi-class prediction:
  - Glioma
  - Meningioma
  - Pituitary Tumor
  - No Tumor
- High accuracy (~97%)
- Explainable AI using Grad-CAM
- Real-time inference via Flask web app
- End-to-end deployment-ready pipeline

---

## 🖼️ Flask Web Application

### 🔹 Home Page – MRI Upload Interface

![Flask App Home](images/front.png)

---

### 🔹 Prediction Output – Tumor Classification Result

![Prediction Result](images/result.png)

> 📌 These screenshots demonstrate real-time MRI upload and model inference using the deployed CNN model.

---

## 🏗️ System Architecture

MRI Image
↓
Image Preprocessing (Resize, Normalize)
↓
CNN Feature Extraction
↓
Softmax Classification
↓
Tumor Type Prediction
↓
Flask Web Interface

yaml
Copy code

---

## 🧠 Model Architecture
- Convolutional layers for feature extraction
- Batch Normalization for training stability
- MaxPooling for spatial reduction
- Dropout to prevent overfitting
- Global Average Pooling
- Dense + Softmax output layer

**Input Size:** 224 × 224 × 3  
**Output Classes:** 4

---

## 📊 Performance
- **Overall Accuracy:** ~97%
- **Inference Time:** < 1 second per image
- **Optimizer:** Adam
- **Loss Function:** Categorical Cross-Entropy

> ⚠️ Recall was prioritized over accuracy to minimize false negatives, which is critical in medical diagnosis.

---

## 🛠️ Tech Stack
- **Language:** Python
- **Deep Learning:** TensorFlow, Keras
- **Image Processing:** OpenCV
- **Web Framework:** Flask
- **Visualization:** Grad-CAM
- **Deployment:** Flask + Render (or Local)

---

## 📁 Project Structure

brain_tumor_detection/
│
├── app.py # Flask app
├── model.py # CNN architecture
├── train.py # Training script
├── predict.py # Prediction utility
├── model/
│ └── brain_tumor.h5
├── templates/
│ └── index.html
├── static/
│ └── uploads/
└── utils/
└── preprocess.py

yaml
Copy code

---

## ▶️ How to Run Locally

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/brain-tumor-detection.git
cd brain-tumor-detection
2️⃣ Install Dependencies
bash
Copy code
pip install -r requirements.txt
3️⃣ Run Flask App
bash
Copy code
python app.py
Open browser at:

cpp
Copy code
http://127.0.0.1:5000
🔍 Explainability (Grad-CAM)
Grad-CAM heatmaps are used to visualize tumor-relevant regions in MRI scans, improving model transparency and trustworthiness.

⚠️ Limitations
Trained on publicly available MRI datasets

Not validated on clinical hospital data

Uses 2D CNN (MRI scans are volumetric)

🔮 Future Improvements
3D CNN for volumetric MRI analysis

Multi-modal learning (MRI + clinical data)

Federated learning for privacy

Hospital-scale validation
