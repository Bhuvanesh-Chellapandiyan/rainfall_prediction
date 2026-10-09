import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import sklearn.compose._column_transformer
from sklearn.impute import SimpleImputer

# ----------------------------------------------------
# 1. Compatibility Patches for scikit-learn
# ----------------------------------------------------
# Fix missing '_RemainderColsList' class in older/newer sklearn
if not hasattr(sklearn.compose._column_transformer, '_RemainderColsList'):
    class _RemainderColsList(list):
        pass
    sklearn.compose._column_transformer._RemainderColsList = _RemainderColsList

# Fix missing '_fill_dtype' attribute on SimpleImputer
def patch_imputer(obj):
    if isinstance(obj, SimpleImputer):
        if not hasattr(obj, '_fill_dtype'):
            obj._fill_dtype = getattr(obj.statistics_, 'dtype', np.dtype('float64'))
    if hasattr(obj, 'named_steps'):
        for step in obj.named_steps.values():
            patch_imputer(step)
    if hasattr(obj, 'transformers_'):
        for item in obj.transformers_:
            if len(item) >= 2:
                patch_imputer(item[1])

# ----------------------------------------------------
# 2. Model Loader Function
# ----------------------------------------------------
@st.cache_resource
def load_model():
    model_path = 'rainfall_prediction_model.pkl'
    try:
        # Attempt loading via joblib first
        loaded_model = joblib.load(model_path)
    except Exception:
        # Fallback to standard pickle if joblib fails
        with open(model_path, 'rb') as f:
            loaded_model = pickle.load(f)
    
    # Apply attribute patch directly to the loaded pipeline
    patch_imputer(loaded_model)
    return loaded_model

# ----------------------------------------------------
# 3. Streamlit Page Setup & Model Invocation
# ----------------------------------------------------
st.set_page_config(
    page_title="Rainfall Prediction",
    page_icon="🌧️",
    layout="centered"
)

try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()

st.title("🌧️ Rainfall Prediction System")
st.write(
    "Enter weather conditions to predict "
    "whether it will rain tomorrow."
)

directions = [
    "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"
]

locations = [
    "Sydney", "Melbourne", "Brisbane", "Perth",
    "Adelaide", "Canberra", "Albury", "Hobart"
]

with st.form("rainfall_form"):
    st.subheader("Location and temperature")
    location = st.selectbox("Location", locations)

    col1, col2 = st.columns(2)
    with col1:
        min_temp = st.number_input("Minimum Temperature (°C)", value=15.0)
        temp_9am = st.number_input("Temperature at 9 AM (°C)", value=18.0)
        humidity_9am = st.number_input("Humidity at 9 AM (%)", 0.0, 100.0, 70.0)
        pressure_9am = st.number_input("Pressure at 9 AM (hPa)", value=1015.0)
        cloud_9am = st.number_input("Cloud at 9 AM (0-9)", 0.0, 9.0, 4.0)

    with col2:
        max_temp = st.number_input("Maximum Temperature (°C)", value=25.0)
        temp_3pm = st.number_input("Temperature at 3 PM (°C)", value=23.0)
        humidity_3pm = st.number_input("Humidity at 3 PM (%)", 0.0, 100.0, 60.0)
        pressure_3pm = st.number_input("Pressure at 3 PM (hPa)", value=1012.0)
        cloud_3pm = st.number_input("Cloud at 3 PM (0-9)", 0.0, 9.0, 4.0)

    st.subheader("Rainfall and wind")
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, value=0.0)
    evaporation = st.number_input("Evaporation (mm)", min_value=0.0, value=5.0)
    sunshine = st.number_input("Sunshine (hours)", min_value=0.0, max_value=24.0, value=7.0)
    wind_gust_dir = st.selectbox("Wind Gust Direction", directions)
    wind_gust_speed = st.number_input("Wind Gust Speed (km/h)", min_value=0.0, value=40.0)
    wind_dir_9am = st.selectbox("Wind Direction at 9 AM", directions)
    wind_dir_3pm = st.selectbox("Wind Direction at 3 PM", directions)
    wind_speed_9am = st.number_input("Wind Speed at 9 AM (km/h)", min_value=0.0, value=15.0)
    wind_speed_3pm = st.number_input("Wind Speed at 3 PM (km/h)", min_value=0.0, value=20.0)
    rain_today = st.selectbox("Did it rain today?", ["No", "Yes"])

    submitted = st.form_submit_button("Predict Rainfall")

if submitted:
    input_data = pd.DataFrame([{
        "Location": location,
        "MinTemp": min_temp,
        "MaxTemp": max_temp,
        "Rainfall": rainfall,
        "Evaporation": evaporation,
        "Sunshine": sunshine,
        "WindGustDir": wind_gust_dir,
        "WindGustSpeed": wind_gust_speed,
        "WindDir9am": wind_dir_9am,
        "WindDir3pm": wind_dir_3pm,
        "WindSpeed9am": wind_speed_9am,
        "WindSpeed3pm": wind_speed_3pm,
        "Humidity9am": humidity_9am,
        "Humidity3pm": humidity_3pm,
        "Pressure9am": pressure_9am,
        "Pressure3pm": pressure_3pm,
        "Cloud9am": cloud_9am,
        "Cloud3pm": cloud_3pm,
        "Temp9am": temp_9am,
        "Temp3pm": temp_3pm,
        "RainToday": rain_today
    }])

    try:
        prediction = int(model.predict(input_data)[0])

        st.subheader("Prediction Result")
        if prediction == 1:
            st.error("🌧️ Rain is predicted for tomorrow.")
        else:
            st.success("☀️ No rain is predicted for tomorrow.")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]
            st.write(f"Predicted rain probability: {probabilities[1]:.1%}")

        st.caption("This is a machine-learning estimate, not an official weather forecast.")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.write(
            "Check that your saved model was trained using the "
            "same feature names and preprocessing pipeline."
        )