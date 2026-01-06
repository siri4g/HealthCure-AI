# Web Integrated Automated Disease Detection Using Artificial Intelligence

## 📌Project Overview

This project presents a **web-based automated disease detection system** that leverages **Artificial Intelligence and Deep Learning** to assist in early and accurate diagnosis of critical diseases. The system integrates multiple deep learning models into a unified web interface, enabling users to upload medical images and receive predictions in real time.

The application focuses on **three major diseases**:

* Brain Tumor
* Alzheimer’s Disease
* Pneumonia

The system is designed to be user-friendly, scalable, and suitable for clinical decision support and academic research.


## 🎯 Objectives

* To automate disease detection using deep learning models
* To provide fast and accurate predictions from medical images
* To integrate multiple disease detection models into a single web platform
* To reduce manual diagnosis effort and support early intervention


## 🧠 Diseases and Models Used

1. Brain Tumor Detection

Model: ResNet50
Input: MRI brain images
Purpose: Classify and detect presence of brain tumors using transfer learning

2. Alzheimer’s Disease Detection

Model: ResNet50
Input: MRI brain scans
Purpose: Identify Alzheimer’s stages by learning disease-specific patterns

3. Pneumonia Detection

Model: VGG16
Input: Chest X-ray images
Purpose: Detect pneumonia by analyzing lung abnormalities



## 🖥️ System Architecture

1. User uploads medical images via web interface
2. Image preprocessing and normalization
3. Deep learning model inference
4. Prediction results displayed on the web page

The backend handles model loading and inference, while the frontend ensures a simple and intuitive user experience.



## 🌐 Web Integration

* **Framework:** Flask / Streamlit (depending on version)
* Enables:

  * Image upload
  * Model selection
  * Real-time prediction
  * Result visualization


## 🛠️ Technologies Used

* **Programming Language:** Python
* **Deep Learning:** TensorFlow, Keras
* **Models:** ResNet50, VGG16, Custom CNN
* **Web Framework:** Flask / Streamlit
* **Libraries:** NumPy, OpenCV, Scikit-learn
* **Development Tools:** Jupyter Notebook, VS Code

---

## ▶️ How to Run the Project

### 1. Install Python

Install **Python 3.10 or above** and ensure *Add Python to PATH* is checked.

### 2. Install Required Libraries

```bash
pip install -r requirements.txt
```

(or manually install TensorFlow, Streamlit/Flask, NumPy, OpenCV)

### 3. Navigate to Project Directory

```bash
cd HealthCure_Full
```

### 4. Run the Application

For Streamlit:

```bash
python -m streamlit run streamlitversion.py
```

For Flask:

```bash
python app.py
```

### 5. Open in Browser

```text
http://localhost:8501   (Streamlit)
http://127.0.0.1:5000   (Flask)
```

---

## 📊 Results

* High accuracy in medical image classification
* Reduced diagnosis time
* Single platform for multiple disease detection

---

## 🔮 Future Enhancements

* Add more diseases and multi-class classification
* Integrate cloud deployment
* Improve UI/UX with dashboards
* Add report generation and patient history tracking

---

## 📚 Conclusion

This project demonstrates the effective use of deep learning in healthcare by integrating multiple disease detection models into a single, web-based system. It highlights how AI can support medical professionals by enabling fast, reliable, and accessible diagnostic tools.


Just say the word.
