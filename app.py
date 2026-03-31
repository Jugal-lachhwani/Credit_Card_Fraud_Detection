import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os

# --- Set up the Web App Page ---
st.set_page_config(
    page_title="Credit Card Fraud Detection System",
    page_icon="🛡️",
    layout="centered"
)

st.title("💳 Real-Time Credit Card Fraud Detection")
st.markdown("""
This application uses a hyperparameter-tuned **XGBoost Classifier** to detect fraudulent transactions 
based on geographical, temporal, and demographic patterns.
""")

# --- Load the Model and Scaler ---
@st.cache_resource
def load_assets():
    # Load Scaler
    scaler = joblib.load('model/robust_scaler.pkl')
    
    # Load XGBoost Model
    model = xgb.XGBClassifier()
    model.load_model('model/xgboost_fraud_model.json')
    
    return scaler, model

try:
    scaler, model = load_assets()
except Exception as e:
    st.error(f"Error loading model assets. Ensure 'robust_scaler.pkl' and 'xgboost_fraud_model.json' are in the 'model/' directory. ({e})")
    st.stop()

st.sidebar.header("Transaction Features")
st.sidebar.info("Enter the transaction details to evaluate its legitimacy.")

# --- Build the Input Form ---
with st.form("prediction_form"):
    st.subheader("Financial & Temporal Details")
    col1, col2 = st.columns(2)
    with col1:
        amt = st.number_input("Transaction Amount ($)", min_value=0.0, max_value=50000.0, value=150.0)
        trans_hour = st.number_input("Transaction Hour (0-23)", min_value=0, max_value=23, value=14, step=1)
        unix_time = st.number_input("Unix Timestamp", min_value=0, value=1371816865, step=1)
    with col2:
        age = st.number_input("Cardholder Age", min_value=18, max_value=120, value=35, step=1)
        city_pop = st.number_input("City Population", min_value=0, value=150000, step=1)

    st.subheader("Geographical Details")
    col3, col4 = st.columns(2)
    with col3:
        lat = st.number_input("User Latitude", value=40.7128)
        long = st.number_input("User Longitude", value=-74.0060)
        distance_km = st.number_input("Distance to Merchant (KM)", min_value=0.0, value=12.5)
    with col4:
        merch_lat = st.number_input("Merchant Latitude", value=40.7306)
        merch_long = st.number_input("Merchant Longitude", value=-73.9866)
    
    st.subheader("Categorical Details (Encoded IDs)")
    st.markdown("*Note: Inputs expect the preprocessed continuous integer IDs.*")
    col5, col6, col7 = st.columns(3)
    with col5:
        category = st.number_input("Category ID", min_value=0, value=4, step=1)
        merchant = st.number_input("Merchant ID", min_value=0, value=102, step=1)
    with col6:
        job = st.number_input("Job ID", min_value=0, value=15, step=1)
        city = st.number_input("City ID", min_value=0, value=85, step=1)
    with col7:
        zip_code = st.number_input("ZIP Code ID", min_value=0, value=12345, step=1)
    
    submit_button = st.form_submit_button(label="🔍 Predict Transaction Legitimacy")

# --- Prediction Logic ---
if submit_button:
    # 1. Arrange inputs in the exact order the scaler expects
    feature_names = ['amt', 'trans_hour', 'category', 'age', 'unix_time', 'merch_lat', 'distance_km', 'merchant', 'lat', 'job', 'merch_long', 'city', 'city_pop', 'zip', 'long']
    
    # We must construct a DataFrame so that the Column Names match exactly what the Scaler expects
    input_df = pd.DataFrame([{
        'amt': amt,
        'trans_hour': trans_hour,
        'category': category,
        'age': age,
        'unix_time': unix_time,
        'merch_lat': merch_lat,
        'distance_km': distance_km,
        'merchant': merchant,
        'lat': lat,
        'job': job,
        'merch_long': merch_long,
        'city': city,
        'city_pop': city_pop,
        'zip': zip_code,
        'long': long
    }])

    # 2. Scale the Input Data
    try:
        scaled_input = scaler.transform(input_df)
    except Exception as e:
        st.error(f"Scaling Error: {e}")
        st.stop()

    # 3. Predict using XGBoost
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]

    # 4. Display Results
    st.markdown("---")
    if prediction == 1:
        st.error("🚨 **FRAUDULENT TRANSACTION DETECTED** 🚨")
        st.markdown(f"**Confidence Score:** `{probability * 100:.2f}%`")
        st.warning("Action: Transaction blocked. An SMS verification code has been dispatched to the cardholder.")
    else:
        st.success("✅ **LEGITIMATE TRANSACTION**")
        st.markdown(f"**Confidence Score (Fraud Risk):** `{probability * 100:.2f}%`")
        st.info("Action: Transaction approved.")
