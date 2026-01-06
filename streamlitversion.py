import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import pickle
import tensorflow as tf
import cv2
import joblib

# ===============================
# Set Page Configuration
# ===============================
st.set_page_config(page_title="AI Disease Detection", layout="wide")

# ===============================
# Load Models
# ===============================

@st.cache_resource
def load_brain_tumor_model():
    model = models.resnet50(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 4)
    state_dict = torch.load(
        r"C:\Users\Vishnu Muppuri\HealthCure_Full\HealthCure_Full\models\resnet50_tumor_model.pth",
        map_location=torch.device('cpu')
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model

@st.cache_resource
def load_pneumonia_model():
    return tf.keras.models.load_model(
        r"C:\Users\Vishnu Muppuri\HealthCure_Full\HealthCure_Full\models\trained.h5"
    )

@st.cache_resource
def load_alzheimer_model():
    return tf.keras.models.load_model(
        r"C:\Users\Vishnu Muppuri\HealthCure_Full\HealthCure_Full\models\fine_tuned_model.keras"
    )

@st.cache_resource
def load_diabetes_model():
    with open(r"C:\Users\Vishnu Muppuri\HealthCure_Full\HealthCure_Full\models\diabetes_neww.sav", 'rb') as f:
        model_data = joblib.load(f)
        if isinstance(model_data, dict):
            return model_data.get('model')
        return model_data

# Load models
brain_tumor_model = load_brain_tumor_model()
pneumonia_model = load_pneumonia_model()
alzheimer_model = load_alzheimer_model()
diabetes_model = load_diabetes_model()

# ===============================
# Preprocessing Functions
# ===============================

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def preprocess_mri(image):
    img = image.resize((128, 128))
    img = np.array(img) / 255.0
    return np.expand_dims(img, axis=0)

# ===============================
# Prediction Functions
# ===============================

def predict_brain_tumor(image):
    image = preprocess_image(image)
    with torch.no_grad():
        outputs = brain_tumor_model(image)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, label = torch.max(probabilities, 0)
    class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
    return class_names[label.item()], confidence.item()

def predict_pneumonia(image):
    if isinstance(image, Image.Image):
        image = np.array(image)

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[-1] == 1:
        image = cv2.merge([image, image, image])

    img = cv2.resize(image, (300, 300))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    if img.shape[-1] != 3:
        img = tf.image.grayscale_to_rgb(tf.convert_to_tensor(img)).numpy()

    prediction = pneumonia_model.predict(img)[0][0]
    label = "Pneumonia" if prediction > 0.5 else "Normal"
    confidence = round(prediction if prediction > 0.5 else 1 - prediction, 4)

    return label, confidence

def predict_alzheimer(image):
    img = np.array(image)
    
    if img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    
    img = cv2.resize(img, (128, 128))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.reshape(1, 128, 128, 1)
    img = img / 255.0
    
    prediction = alzheimer_model.predict(img)[0]
    
    class_names = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']
    label_idx = np.argmax(prediction)
    confidence = prediction[label_idx]
    
    return class_names[label_idx], confidence

def predict_diabetes(features):
    features = np.array(features).reshape(1, -1)
    prediction = diabetes_model.predict(features)[0]
    label = "Diabetic" if prediction == 1 else "Non-Diabetic"
    
    if hasattr(diabetes_model, "predict_proba"):
        confidence = diabetes_model.predict_proba(features)[0][prediction]
    else:
        confidence = 1.0
    
    return label, confidence

# ===============================
# Streamlit Interface
# ===============================

def show_prediction_result(label, confidence):
    st.success(f"**Prediction:** {label}")
    st.info(f"**Confidence:** {confidence * 100:.2f}%")

def display_uploaded_image(image):
    st.image(image, caption="Uploaded Image", use_column_width=True)

# ===============================
# Main App
# ===============================

def main_app():
    st.title("Web Integrated Automated Disease Detection")
    st.markdown("### Diagnose Brain Tumor, Pneumonia, Alzheimer's, and Diabetes")

    option = st.sidebar.selectbox(
        "Select Disease to Diagnose", 
        ("Brain Tumor", "Pneumonia", "Alzheimer's", "Diabetes")
    )

    if option in ["Brain Tumor", "Pneumonia", "Alzheimer's"]:
        uploaded_file = st.file_uploader(f"Upload {option} Scan", type=["jpg", "png", "jpeg"])

        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB" if option != "Pneumonia" else "L")
            display_uploaded_image(image)

            if st.button("Predict"):
                if option == "Brain Tumor":
                    label, confidence = predict_brain_tumor(image)
                elif option == "Pneumonia":
                    label, confidence = predict_pneumonia(image)
                elif option == "Alzheimer's":
                    label, confidence = predict_alzheimer(image)

                show_prediction_result(label, confidence)

    elif option == "Diabetes":
        col1, col2 = st.columns(2)

        with col1:
            pregnancies = st.number_input("Number of Pregnancies", 0, 20, 0)
            glucose = st.number_input("Glucose Level", 0, 300, 100)
            blood_pressure = st.number_input("Blood Pressure", 0, 200, 70)
            skin_thickness = st.number_input("Skin Thickness", 0, 100, 20)

        with col2:
            insulin = st.number_input("Insulin Level", 0, 900, 80)
            bmi = st.number_input("BMI", 0.0, 60.0, 25.0)
            diabetes_pedigree = st.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
            age = st.number_input("Age", 0, 120, 30)

        if st.button("Predict"):
            features = [
                pregnancies, glucose, blood_pressure, 
                skin_thickness, insulin, bmi, diabetes_pedigree, age
            ]
            label, confidence = predict_diabetes(features)
            show_prediction_result(label, confidence)

# ===============================
# Run App
# ===============================
if __name__ == "__main__":
    main_app()
