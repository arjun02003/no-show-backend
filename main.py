from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(title="No-Show Prediction API")

# Load trained model & frozen feature list
model = joblib.load("xgboost_no_show_model.pkl")
model_features = joblib.load("model_features.pkl")


# ---------- INPUT SCHEMA ----------
# Sirf BASIC features user se lenge
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
    return {"status": "No-Show Prediction API is running"}


# ---------- PREDICTION ----------
@app.post("/predict")
def predict(data: Patient):

    # Step 1: user input dataframe
    df = pd.DataFrame([data.dict()])

    # Step 2: add ALL missing one-hot columns as 0
    for col in model_features:
        if col not in df.columns:
            df[col] = 0

    # Step 3: reorder columns EXACTLY as training
    df = df[model_features]

    # Step 4: predict
    prob = model.predict_proba(df)[0][1]

    return {
        "no_show_risk": round(float(prob), 3)
    }


# ---------- DEBUG ----------
@app.get("/debug")
def debug():
    return {
        "model_loaded": True,
        "total_model_features": len(model_features),
        "sample_features": model_features[:15]
    }
