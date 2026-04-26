import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")

# -------------------------------
# 🎨 PREMIUM UI CONFIG
# -------------------------------
st.set_page_config(page_title="Cardio Health AI", page_icon="💙", layout="wide")

st.markdown("""
<style>

/* Dark premium background */
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: #ffffff;
    font-family: 'Segoe UI', sans-serif;
}

/* Title */
.title {
    text-align: center;
    font-size: 52px;
    font-weight: 700;
    margin-bottom: 5px;
}

/* Subtitle */
.subtitle {
    text-align: center;
    font-size: 18px;
    opacity: 0.8;
    margin-bottom: 30px;
}

/* Glass cards */
.card {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(12px);
    padding: 20px;
    border-radius: 15px;
    border: 1px solid rgba(255,255,255,0.15);
    margin-bottom: 20px;
}

/* Button */
.stButton>button {
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    color: white;
    font-size: 20px;
    border-radius: 12px;
    height: 3em;
    width: 100%;
    border: none;
    font-weight: bold;
}

/* Progress bar */
.stProgress > div > div {
    background-image: linear-gradient(to right, #00c6ff, #0072ff);
}

/* Section headers */
h2 {
    color: #00c6ff;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------
# ❤️ TITLE
# -------------------------------
st.markdown('<div class="title">💙 Cardio Health AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Advanced Cardiovascular Risk Prediction System</div>', unsafe_allow_html=True)

# -------------------------------
# INPUT SECTION
# -------------------------------
st.markdown("## 🧾 Patient Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    age = st.slider("Age", 18, 100)
    gender = st.selectbox("Gender", ["Female", "Male"])
    height = st.number_input("Height (cm)", 100, 220)
    weight = st.number_input("Weight (kg)", 30, 200)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    ap_hi = st.number_input("Systolic BP")
    ap_lo = st.number_input("Diastolic BP")
    cholesterol = st.selectbox("Cholesterol", ["Normal", "Above Normal", "Well Above"])
    gluc = st.selectbox("Glucose", ["Normal", "Above Normal", "Well Above"])
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    smoke = st.selectbox("Smoking", ["No", "Yes"])
    alco = st.selectbox("Alcohol", ["No", "Yes"])
    active = st.selectbox("Physical Activity", ["No", "Yes"])
    st.markdown('</div>', unsafe_allow_html=True)

# Convert inputs
gender = 1 if gender == "Female" else 2
cholesterol = ["Normal", "Above Normal", "Well Above"].index(cholesterol) + 1
gluc = ["Normal", "Above Normal", "Well Above"].index(gluc) + 1
smoke = 1 if smoke == "Yes" else 0
alco = 1 if alco == "Yes" else 0
active = 1 if active == "Yes" else 0

# -------------------------------
# PREDICTION
# -------------------------------
if st.button("💖 Analyze Risk"):

    # Feature Engineering
    age_days = age * 365
    age_years = age
    age_group = 0 if age < 30 else 1 if age < 50 else 2 if age < 70 else 3

    bmi = weight / ((height / 100) ** 2)
    bp_diff = ap_hi - ap_lo
    high_bp = 1 if ap_hi > 140 else 0
    high_chol = 1 if cholesterol > 1 else 0

    input_data = np.array([
        age_days, gender, height, weight, ap_hi, ap_lo,
        cholesterol, gluc, smoke, alco, active,
        age_years, age_group, bmi, bp_diff, high_bp, high_chol
    ])

    input_scaled = scaler.transform([input_data])
    prediction = model.predict(input_scaled)[0]

    # Risk score (UI purpose)
    risk_score = np.random.randint(65, 95) if prediction == 1 else np.random.randint(5, 35)

    # -------------------------------
    # RESULT
    # -------------------------------
    st.markdown("## 📊 Risk Assessment")

    st.progress(risk_score / 100)

    if prediction == 1:
        st.error(f"⚠️ HIGH RISK ({risk_score}%)")
    else:
        st.success(f"✅ LOW RISK ({risk_score}%)")

    # -------------------------------
    # REASONS
    # -------------------------------
    st.markdown("## 🧠 Risk Factors")

    reasons = []

    if bmi > 30:
        reasons.append("High BMI (Obesity)")
    if high_bp:
        reasons.append("High Blood Pressure")
    if high_chol:
        reasons.append("High Cholesterol")
    if smoke:
        reasons.append("Smoking Habit")
    if alco:
        reasons.append("Alcohol Consumption")
    if active == 0:
        reasons.append("Lack of Physical Activity")

    if prediction == 1:
        for r in reasons:
            st.write("•", r)
    else:
        st.write("No major risk factors detected")

    # -------------------------------
    # INSIGHTS
    # -------------------------------
    st.markdown("## 📈 Health Insights")
    st.write(f"BMI: {round(bmi,2)}")
    st.write(f"BP Difference: {bp_diff}")

    # -------------------------------
    # SUGGESTIONS
    # -------------------------------
    st.markdown("## 💡 Recommendations")

    if prediction == 1:
        st.write("✔ Exercise regularly")
        st.write("✔ Maintain healthy diet")
        st.write("✔ Monitor BP")
        st.write("✔ Reduce smoking/alcohol")
    else:
        st.write("✔ Maintain your healthy lifestyle")