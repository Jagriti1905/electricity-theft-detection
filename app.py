import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load model and dataset
# -----------------------------
model = joblib.load("electricity_theft_model.pkl")
df = pd.read_csv("data/cleaned_data.csv")

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Electricity Theft Detection",
    page_icon="⚡",
    layout="wide"
)

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>
.main {
    padding-top: 2rem;
}

.title {
    font-size: 42px;
    font-weight: bold;
}

.subtitle {
    font-size: 18px;
    color: #888888;
}

.info-card {
    padding: 20px;
    border-radius: 12px;
    background-color: #1e1e1e;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
st.markdown(
    '<div class="title">⚡ Electricity Theft Detection</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Machine Learning based electricity consumption analysis</div>',
    unsafe_allow_html=True
)

st.divider()

# -----------------------------
# Project information
# -----------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Consumers", len(df))

with col2:
    st.metric("ML Model", "Random Forest")

with col3:
    st.metric("Features", df.shape[1] - 2)

st.divider()

# -----------------------------
# Prediction section
# -----------------------------
st.subheader("🔍 Consumer Prediction")

st.write(
    "Select a consumer row and use the trained machine learning model "
    "to identify whether the consumption pattern appears normal or potentially suspicious."
)

consumer_index = st.number_input(
    "Enter Consumer Row Number",
    min_value=0,
    max_value=len(df) - 1,
    value=0,
    step=1
)

if st.button("🚀 Predict", use_container_width=True):

    # Remove target and consumer ID
    X = df.drop(columns=["FLAG", "CONS_NO"])

    # Select consumer
    sample = X.iloc[[consumer_index]]

    # Prediction
    prediction = model.predict(sample)[0]

    # Probability
    probability = model.predict_proba(sample)[0][1]

    st.divider()

    # -----------------------------
    # Display result
    # -----------------------------
    if prediction == 1:

        st.error("⚠️ Potentially Suspicious Consumption")

        st.metric(
            "Suspicious Probability",
            f"{probability * 100:.2f}%"
        )

        st.warning(
            "This is a machine learning prediction and does not confirm "
            "actual electricity theft."
        )

    else:

        st.success("✅ Normal Consumption")

        st.metric(
            "Suspicious Probability",
            f"{probability * 100:.2f}%"
        )

        st.info(
            "The model considers this consumption pattern to be normal."
        )

    # -----------------------------
    # Consumer information
    # -----------------------------
    st.subheader("📋 Consumer Information")

    info_col1, info_col2 = st.columns(2)

    with info_col1:
        st.write("**Consumer Row:**", consumer_index)

    with info_col2:
        st.write("**Prediction:**", "Suspicious" if prediction == 1 else "Normal")

st.divider()

st.caption(
    "⚠️ This system is designed for educational and analytical purposes. "
    "Predictions should be verified using appropriate utility and field-level information."
)
