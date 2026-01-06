# 🏥 Web Integrated Automated Disease Detection Using Artificial Intelligence

## 📌 Project Overview

This project delivers a **web-based, AI-powered medical diagnosis system** designed to assist in the early detection of critical diseases using medical imaging. By integrating multiple deep learning models into a single web application, the system enables fast, accurate, and accessible disease prediction through a simple user interface.

The solution focuses on **three major diseases**:

* **Brain Tumor**
* **Alzheimer’s Disease**
* **Pneumonia**

Each disease is handled using a model architecture chosen for its proven effectiveness in medical image analysis.

---

## 🎯 Objectives

* Automate disease detection using deep learning
* Improve diagnostic accuracy and speed
* Provide a unified web platform for multiple diseases
* Reduce manual effort and support clinical decision-making

---

## 🧠 Disease Modules & Models

### 🧬 Brain Tumor Detection

* **Model:** ResNet50 (Transfer Learning)
* **Input:** MRI brain images
* **Description:**
  A deep residual network is fine-tuned to extract high-level spatial features from MRI scans, enabling robust tumor detection with reduced vanishing-gradient issues.

---

### 🧠 Alzheimer’s Disease Detection

* **Model:** ResNet50 (Transfer Learning)
* **Input:** MRI brain scans
* **Description:**
  The same residual learning architecture is adapted to detect Alzheimer’s-related structural changes in the brain, providing reliable classification across disease stages.

---

### 🫁 Pneumonia Detection

* **Model:** Custom Convolutional Neural Network (CNN)
* **Input:** Chest X-ray images
* **Description:**
  A lightweight CNN is designed and trained from scratch to capture lung texture patterns, enabling effective pneumonia detection with optimized computational cost.

---

## 🖥️ System Architecture

1. User uploads medical images via web interface
2. Image preprocessing (resizing, normalization)
3. Disease-specific deep learning model inference
4. Prediction result displayed instantly

The modular design allows each disease model to operate independently while sharing a common web interface.

---

## 🌐 Web Integration

* **Framework:** Flask / Streamlit
* Features:

  * Image upload
  * Disease selection
  * Real-time predictions
  * Clean and interactive UI

This integration ensures ease of use for both technical and non-technical users.

---

## 🛠️ Technologies Used

* **Programming Language:** Python
* **Deep Learning:** TensorFlow, Keras
* **Models:** ResNet50, Custom CNN
* **Web Framework:** Flask / Streamlit
* **Libraries:** NumPy, OpenCV, Scikit-learn
* **Tools:** Jupyter Notebook, VS Code

---

## ▶️ How to Run the Project

### 1️⃣ Install Python

Install **Python 3.10 or higher**
✔️ Ensure **Add Python to PATH** is enabled

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Navigate to Project Folder

```bash
cd HealthCure_Full
```

### 4️⃣ Run the Application

**Streamlit version**

```bash
python -m streamlit run streamlitversion.py
```

**Flask version**

```bash
python app.py
```

### 5️⃣ Open in Browser

* Streamlit → `http://localhost:8501`
* Flask → `http://127.0.0.1:5000`

---

## 📊 Results

* High classification accuracy across all three diseases
* Fast inference time
* Unified and scalable disease detection platform

---

## 🔮 Future Scope

* Multi-class severity classification
* Cloud deployment (AWS / GCP)
* Integration with electronic health records (EHR)
* Explainable AI (Grad-CAM visualizations)

---

## 📚 Conclusion

This project demonstrates the practical application of **deep learning in healthcare** by integrating **ResNet50-based transfer learning and custom CNN architectures** into a single web-based diagnostic system. It showcases how AI can enhance medical decision-making through speed, accuracy, and accessibility.

---
