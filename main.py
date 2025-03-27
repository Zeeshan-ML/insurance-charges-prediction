from fastapi import FastAPI # type: ignore
import joblib
import pandas as pd
import numpy as np
from pydantic import BaseModel

# Load models and scaler
lr_model = joblib.load("model/lr_model.joblib")
knn_model = joblib.load("model/knn_model.joblib")
scaler = joblib.load("model/scaler.joblib")
feature_columns = joblib.load("model/feature_columns.joblib")
num_features = ['age', 'bmi']
# Initialize FastAPI app
app = FastAPI()
# Define input data schema
class InputData(BaseModel):
    age: int
    sex: str
    bmi: float
    children: int
    smoker: str
    region: str

# Prediction function
def predict(new_data: dict):
    df = pd.DataFrame([new_data])
    df['smoker'] = df["smoker"].map({"yes": 1, "no": 0})
    df["sex"] = df["sex"].map({"male": 1, "female": 0})
    df = pd.get_dummies(df, drop_first=True, dtype=int)
    df = df.reindex(columns=feature_columns, fill_value=0)
    df[num_features] = scaler.transform(df[num_features])
    prediction = np.expm1(lr_model.predict(df))
    return float(prediction[0])

# API endpoint
@app.post("/predict")
def get_prediction(data: InputData):
    result = predict(data.model_dump())
    return {"predicted_charges": result}
