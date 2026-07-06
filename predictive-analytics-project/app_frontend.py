# app_frontend.py
import streamlit as st
import requests

# Page Styling
st.set_page_config(page_title="AI Predictive Dashboard", page_icon="📊", layout="centered")

st.title("📊 Healthcare Charges Predictive Dashboard")
st.markdown("---")
st.write("Input patient demographics below to generate an AI-driven medical cost forecast.")

# Layout Columns
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Patient Age", min_value=18, max_value=100, value=30, step=1)
    bmi = st.slider("BMI (Body Mass Index)", min_value=10.0, max_value=60.0, value=24.5, step=0.1)

with col2:
    smoker_label = st.selectbox("Smoking Status", options=["No", "Yes"])
    # Convert text to binary model expected input (0 or 1)
    smoker = 1 if smoker_label == "Yes" else 0

st.markdown("---")

# Prediction trigger button
if st.button("🚀 Calculate Predicted Cost", use_container_width=True):
    # Construct payload match backend expectation precisely
    payload = {
        "age": age,
        "bmi": bmi,
        "smoker": smoker
    }
    
    BACKEND_URL = "http://127.0.0.1:8000/predict"
    
    try:
        with st.spinner("Communicating with FastAPI AI Model Server..."):
            response = requests.post(BACKEND_URL, json=payload)
            
        if response.status_code == 200:
            result = response.json()
            predicted_cost = result["predicted_charges_usd"]
            
            # Show output metrics dynamically
            st.success("Prediction Completed Successfully!")
            st.metric(label="Estimated Annual Medical Charges", value=f"${predicted_cost:,.2f}")
            
            # Contextual alert banner
            if smoker == 1:
                st.warning("⚠️ High premium warning applied due to smoker status flag.")
        else:
            st.error(f"Backend Error (Status {response.status_code}): {response.text}")
            
    except requests.exceptions.ConnectionError:
        st.error("Error: Could not connect to the Backend server. Ensure app_backend.py is running on port 8000.")