import sys
import os

# --- Fix for ModuleNotFoundError ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image

from preprocess import load_preprocessor
from utils import read_image_from_upload, load_mobilenet_embed_model, image_to_embedding, map_prob_to_label

# --- Streamlit page config ---
st.set_page_config(page_title="Child Nutrition Risk Detector", layout="centered")
st.title("Child Nutrition Risk Detector")
st.write("Enter the child details (20 parameters), upload photo, then click Predict")

# --- Streamlit form for input ---
with st.form("input_form"):
    Age = st.number_input("Age (months)", min_value=0, max_value=240, value=24, step=1)
    Weight = st.number_input("Weight (kg)", min_value=0.1, value=10.0, step=0.1)
    Height = st.number_input("Height (cm)", min_value=10.0, value=80.0, step=0.1)  # float defaults
    BMI = st.number_input("BMI (optional) - leave 0 if unknown", min_value=0.0, value=0.0, step=0.1)
    Gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    Mother_Edu = st.text_input("Mother's Education (primary, secondary, higher)")
    Father_Edu = st.text_input("Father's Education (primary, secondary, higher)")
    Household_Income = st.number_input("Household Income (monthly)", min_value=0, step=100)
    Meals_per_Day = st.number_input("Meals per Day", min_value=0, max_value=10, value=3, step=1)
    Vaccination_Status = st.selectbox("Vaccination Status", ["complete", "partial", "none"])
    Access_Clean_Water = st.selectbox("Access to Clean Water", ["yes", "no"])
    Region = st.text_input("Region / Area")
    Birth_Weight = st.number_input("Birth Weight (kg)", min_value=0.1, value=2.5, step=0.1)
    Family_Size = st.number_input("Family Size", min_value=1, value=4, step=1)
    Food_Habits = st.text_input("Food Habits (vegetarian, mixed)")
    Inherited_Diseases = st.text_input("Inherited Diseases (comma-separated)")
    Appetite_Level = st.slider("Appetite Level (1-5)", 1, 5, 3)
    Place_of_Birth = st.text_input("Place of Birth (hospital/home)")
    Sanitation_Access = st.selectbox("Sanitation Access", ["yes", "no"])
    Breastfeeding_Duration = st.number_input("Breastfeeding Duration (months)", min_value=0, value=0, step=1)

    # optional blood parameters
    st.markdown("**Optional blood parameters (leave blank if unknown)**")
    Hemoglobin = st.number_input("Hemoglobin (g/dL)", min_value=0.0, value=0.0, step=0.1)
    Ferritin = st.number_input("Ferritin (ng/mL)", min_value=0.0, value=0.0, step=0.1)
    CRP = st.number_input("CRP (mg/L)", min_value=0.0, value=0.0, step=0.1)

    uploaded_file = st.file_uploader("Upload child photo (JPEG/PNG)", type=["jpg","jpeg","png"])
    submitted = st.form_submit_button("Predict")

# --- On form submit ---
if submitted:
    # Create dataframe for preprocessing
    input_dict = {
        "Age": Age, "Weight": Weight, "Height": Height,
        "BMI": BMI if BMI > 0 else None,
        "Gender": Gender, "Mother's Education": Mother_Edu, "Father's Education": Father_Edu,
        "Household Income (monthly)": Household_Income, "Meals per Day": Meals_per_Day,
        "Vaccination Status": Vaccination_Status, "Access to Clean Water": Access_Clean_Water,
        "Region / Area": Region, "Birth Weight (kg)": Birth_Weight, "Family Size": Family_Size,
        "Food Habits": Food_Habits, "Inherited Diseases": Inherited_Diseases,
        "Appetite Level": Appetite_Level, "Place of Birth": Place_of_Birth,
        "Sanitation Access": Sanitation_Access, "Breastfeeding Duration (months)": Breastfeeding_Duration,
        "Hemoglobin": Hemoglobin, "Ferritin": Ferritin, "CRP": CRP
    }

    st.subheader("Inputs (as form)")
    st.json(input_dict)

    # --- Load preprocessor and model ---
    preproc_path = os.path.join(current_dir, "../models/preprocessor.joblib")
    model_path = os.path.join(current_dir, "../models/ensemble_model.joblib")

    if not os.path.exists(preproc_path) or not os.path.exists(model_path):
        st.error("Model or preprocessor not found. Please run training scripts first.")
    else:
        preprocessor, num_features, cat_features = load_preprocessor(preproc_path)
        clf = joblib.load(model_path)

        df_input = pd.DataFrame([input_dict])
        X_tab = preprocessor.transform(df_input[num_features + cat_features])

        # Add image embedding if uploaded
        if uploaded_file is not None:
            img = read_image_from_upload(uploaded_file)
            embed_model = load_mobilenet_embed_model()
            emb = image_to_embedding(img, embed_model)
            X = np.hstack([X_tab, emb.reshape(1, -1)])
        else:
            X = X_tab

        # Predict
        probs = clf.predict_proba(X)[0]
        pred_idx = int(np.argmax(probs))
        classes = clf.classes_ if hasattr(clf, "classes_") else ["low", "moderate", "high"]
        pred_label = classes[pred_idx]

        # Color mapping
        color_map = {"low": "green", "moderate": "orange", "high": "red"}
        color = color_map.get(pred_label, "black")

        st.markdown(f"<h2 style='color:{color}'>Predicted risk: {pred_label.upper()}</h2>", unsafe_allow_html=True)
        st.write("Class probabilities:")
        st.write({classes[i]: float(probs[i]) for i in range(len(probs))})
