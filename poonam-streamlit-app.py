# app.py
import streamlit as st
import joblib
import pandas as pd

# ---------- Page ----------
st.set_page_config(
    page_title="Bankruptcy Prediction App",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ---------- App Title ----------
st.markdown("<h1 style='text-align: center; color: #2E86C1;'>📊 Bankruptcy Prediction App</h1>", unsafe_allow_html=True)
st.write("### 💼 Predict whether a company is likely to go bankrupt based on key financial risk factors.")

st.markdown("---")

# ---------- Load the saved model ----------
model = joblib.load("final_logistic_model.pkl")

# ---------- Create Two Columns for Input ----------
col1, col2 = st.columns(2)

with col1:
    industrial = st.selectbox("🏭 Industrial Risk", [0, 0.5, 1.0])
    management = st.selectbox("👔 Management Risk", [0, 0.5, 1.0])
    financial = st.selectbox("💰 Financial Flexibility", [0, 0.5, 1.0])

with col2:
    credibility = st.selectbox("🤝 Credibility", [0, 0.5, 1.0])
    competitiveness = st.selectbox("📈 Competitiveness", [0, 0.5, 1.0])
    operating = st.selectbox("⚙️ Operating Risk", [0, 0.5, 1.0])

st.markdown("---")

# ---------- Predict Button ----------
if st.button("🚀 Predict Bankruptcy Status", use_container_width=True):
    # Create DataFrame for new input
    X_new = pd.DataFrame([[industrial, management, financial, credibility, competitiveness, operating]],
                         columns=['industrial_risk','management_risk','financial_flexibility','credibility','competitiveness','operating_risk'])
    
    # Make prediction
    pred = model.predict(X_new)[0]
    proba = model.predict_proba(X_new)[0,1]
    
    # Display Results
    st.markdown("### 🧠 Prediction Result")
    if pred == 1:
        st.error("⚠️ The model predicts **BANKRUPT**.")
    else:
        st.success("✅ The model predicts **NON-BANKRUPT**.")
    
    st.markdown(f"**Probability of Bankruptcy:** `{proba:.2f}`")
    st.progress(proba)

    if proba > 0.7:
        st.warning("⚠️ High probability! The company is at serious financial risk.")
    elif proba > 0.4:
        st.info("ℹ️ Moderate probability. The company may need close monitoring.")
    else:
        st.balloons()
        st.success("🎉 Low probability! The company appears financially stable.")

st.markdown("---")
st.caption("Developed by Poonam Palekar | Logistic Regression Model | Streamlit App")
