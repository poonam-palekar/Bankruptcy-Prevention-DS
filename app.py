# app.py
import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Bankruptcy Prediction App", page_icon="📊", layout="centered")

st.title("Bankruptcy Prediction App")
st.write("Enter company risk factors below to predict whether it is likely to go bankrupt.")

# Load the saved model
model = joblib.load("final_logistic_model.pkl")

# Input fields for all 6 features
industrial = st.selectbox("Industrial Risk", [0, 0.5, 1.0])
management = st.selectbox("Management Risk", [0, 0.5, 1.0])
financial = st.selectbox("Financial Flexibility", [0, 0.5, 1.0])
credibility = st.selectbox("Credibility", [0, 0.5, 1.0])
competitiveness = st.selectbox("Competitiveness", [0, 0.5, 1.0])
operating = st.selectbox("Operating Risk", [0, 0.5, 1.0])

if st.button("Predict"):
    # Create dataframe for new input
    X_new = pd.DataFrame([[industrial, management, financial, credibility, competitiveness, operating]],
                         columns=['industrial_risk','management_risk','financial_flexibility','credibility','competitiveness','operating_risk'])
    
    # Prediction
    pred = model.predict(X_new)[0]
    proba = model.predict_proba(X_new)[0,1]
    
    # Display results
    st.subheader("📈 Prediction Result:")
    st.write("Prediction:", "🔴 **Bankrupt**" if pred==1 else "🟢 **Non-bankrupt**")
    st.progress(proba)
    st.write(f"**Probability of Bankruptcy:** {proba:.2f}")
