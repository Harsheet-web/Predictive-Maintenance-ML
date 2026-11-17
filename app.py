import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(page_title="Predictive Maintenance", page_icon="⚙️")
st.title("Predictive Maintenance System")
st.caption("AI4I 2020 — educational machine-failure prediction demo")

MODEL_PATH = Path("predictive_maintenance_model.pkl")
SCALER_PATH = Path("scaler.pkl")
ENCODER_PATH = Path("encoder.pkl")

missing = [p.name for p in [MODEL_PATH, SCALER_PATH, ENCODER_PATH] if not p.exists()]
if missing:
    st.error("Missing model artifacts: " + ", ".join(missing))
    st.info("Export these files from your training notebook into the same folder as app.py.")
    st.stop()

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
encoder = joblib.load(ENCODER_PATH)

st.subheader("Machine parameters")
machine_type = st.selectbox("Machine Type", ["L", "M", "H"])
air_temp = st.number_input("Air temperature [K]", value=300.0, step=0.1)
process_temp = st.number_input("Process temperature [K]", value=310.0, step=0.1)
speed = st.number_input("Rotational speed [rpm]", min_value=0, value=1500, step=10)
torque = st.number_input("Torque [Nm]", min_value=0.0, value=40.0, step=0.5)
wear = st.number_input("Tool wear [min]", min_value=0, value=100, step=1)

if st.button("Predict machine failure", type="primary"):
    temp_diff = process_temp - air_temp
    mechanical_power = torque * (2 * np.pi * speed / 60)

    # Keep this numeric column order identical to the training notebook.
    numeric_input = pd.DataFrame({
        "Air temperature [K]": [air_temp],
        "Process temperature [K]": [process_temp],
        "Rotational speed [rpm]": [speed],
        "Torque [Nm]": [torque],
        "Tool wear [min]": [wear],
        "Temperature Difference": [temp_diff],
        "Mechanical Power": [mechanical_power],
    })

    categorical_input = pd.DataFrame({"Type": [machine_type]})
    numeric_scaled = scaler.transform(numeric_input)
    type_encoded = encoder.transform(categorical_input)
    model_input = np.concatenate([numeric_scaled, type_encoded], axis=1)

    prediction = int(model.predict(model_input)[0])
    failure_probability = float(model.predict_proba(model_input)[0, 1])

    st.subheader("Prediction")
    if prediction == 1:
        st.error("Machine failure predicted")
    else:
        st.success("No machine failure predicted")

    st.metric("Estimated failure probability", f"{failure_probability:.1%}")
    st.caption("This is an experimental model estimate, not a safety guarantee.")
