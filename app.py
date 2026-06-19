import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- 1. Load the Model & Helpers ---
@st.cache_resource
def load_data():
    model = joblib.load('bmw_price_model.pkl')
    model_columns = joblib.load('model_features.pkl')
    # Load raw data just to get lists for dropdowns (unique values)
    # Assuming you have the CSV in the same folder
    df = pd.read_csv('BMW_Sales_Cleaned_Data.csv') 
    return model, model_columns, df

model, model_columns, df = load_data()

# --- 2. App Layout & Title ---
st.set_page_config(page_title="BMW Price Predictor", page_icon="🚗", layout="centered")

st.image("https://upload.wikimedia.org/wikipedia/commons/4/44/BMW.svg", width=100)
st.title("BMW Used Car Price Predictor")
st.markdown("Enter the car details below to get an estimated market price using AI.")
st.markdown("---")

# --- 3. User Inputs (Side by Side) ---
col1, col2 = st.columns(2)

with col1:
    car_model = st.selectbox("Select Model", df['Model'].unique())
    fuel_type = st.selectbox("Fuel Type", df['Fuel_Type'].unique())
    transmission = st.selectbox("Transmission", df['Transmission'].unique())

with col2:
    year = st.slider("Manufacturing Year", 2010, 2024, 2018)
    mileage = st.number_input("Mileage (KM)", min_value=0, value=50000, step=1000)
    engine_size = st.slider("Engine Size (L)", 1.5, 5.0, 2.0, step=0.1)

# Hidden input for Sales_Volume (using median as default since it's an aggregate metric)
sales_vol = df['Sales_Volume'].median()

# --- 4. Prediction Logic ---
if st.button(" Predict Price", type="primary"):
    
    # A. Prepare the input data
    # Calculate derived features
    car_age = 2025 - year
    mileage_log = np.log1p(mileage)
    
    # Note: We must apply the SAME Scaling as training. 
    # For simplicity in this demo, we assume raw values or we should load the scaler too.
    # Ideally, save 'scaler' in Step 1 and load it here.
    # For now, let's proceed (Trees are somewhat robust, but scaling is better).
    
    input_data = pd.DataFrame({
        'Car_Age': [car_age],
        'Engine_Size_L': [engine_size],
        'Mileage_Log': [mileage_log],
        'Sales_Volume': [sales_vol],
        'Model': [car_model],
        'Fuel_Type': [fuel_type],
        'Transmission': [transmission]
    })
    
    # B. One-Hot Encoding (Align with Training Data)
    input_encoded = pd.get_dummies(input_data)
    # Align columns: Add missing columns with 0, remove extra ones
    input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)
    
    # C. Make Prediction
    prediction = model.predict(input_encoded)[0]
    
    # D. Display Result
    st.success(f"Estimated Price: ${prediction:,.2f}")
    
    # Interpretation Helper
    if prediction > 72000:
        st.balloons()
        st.info("This is a high-value luxury car!")
    elif prediction < 650000:
        st.info("This is a budget-friendly option.")

# --- Footer ---
st.markdown("---")
st.caption("Powered by Bishoy Safwat | QA Project Demo")