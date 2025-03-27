import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load models and scaler
lr_model = joblib.load("model/lr_model.joblib")
knn_model = joblib.load("model/knn_model.joblib")
scaler = joblib.load("model/scaler.joblib")
feature_columns = joblib.load("model/feature_columns.joblib")
num_features = ['age', 'bmi']

def predict(new_data: dict):
    df = pd.DataFrame([new_data])
    df['smoker'] = df["smoker"].map({"yes": 1, "no": 0})
    df["sex"] = df["sex"].map({"male": 1, "female": 0})
    df = pd.get_dummies(df, drop_first=True, dtype=int)
    df = df.reindex(columns=feature_columns, fill_value=0)
    df[num_features] = scaler.transform(df[num_features])
    prediction = np.expm1(lr_model.predict(df))
    return float(prediction[0])

# Streamlit UI
st.title("Medical Insurance Charges Prediction")

# Input Fields
age = st.number_input("Age",value=30)
sex = st.selectbox("Sex", ["male", "female"])
bmi = st.number_input("BMI",value=25.0, format="%.1f")
children = st.number_input("Number of Children", value=1)
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

# Prediction Button
if st.button("Predict Charges"):
    user_data = {"age": age, "sex": sex, "bmi": bmi, "children": children, "smoker": smoker, "region": region}
    result = predict(user_data)
    st.success(f"Predicted Medical Insurance Charges: ${result:.2f}")
