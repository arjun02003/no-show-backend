from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# Load trained model and feature list
model = joblib.load("xgboost_no_show_model.pkl")
model_features = joblib.load("model_features.pkl")

# Only SIMPLE inputs from user
class Patient(BaseModel):
    Age: int
    LeadTimeDays: int
    ScheduledHour: int
    AppointmentDayOfWeek: int
    SMS_received: int
    PastNoShows: int
    Scholarship: int
    Hipertension: int
    Diabetes: int
    Alcoholism: int
    Handcap: int

@app.get("/")
def home():
    return {"status": "No-Show Prediction API running"}

@app.post("/predict")
def predict(data: Patient):
    # user input
    input_data = data.dict()

    # create empty row with ALL model features = 0
    row = {feature: 0 for feature in model_features}

    # fill only available features
    for key, value in input_data.items():
        if key in row:
            row[key] = value

    # create dataframe in exact order
    df = pd.DataFrame([row], columns=model_features)

    # prediction
    prob = model.predict_proba(df)[0][1]

    return {
        "no_show_risk": round(float(prob), 3)
    }

@app.get("/debug")
def debug():
    return {
        "model_loaded": True,
        "features_count": len(model_features),
        "first_10_features": model_features[:10]
    }
