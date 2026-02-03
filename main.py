from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# load model + frozen feature list
model = joblib.load("xgboost_no_show_model.pkl")
model_features = joblib.load("model_features.pkl")


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
    Handcap: int   # ✅ MUST match training feature name


@app.get("/")
def home():
    return {"status": "No-Show Prediction API is running"}


@app.post("/predict")
def predict(data: Patient):
    # convert input to dataframe
    df = pd.DataFrame([data.dict()])

    # add missing columns
    for col in model_features:
        if col not in df.columns:
            df[col] = 0

    # ensure correct column order
    df = df[model_features]

    # predict probability
    prob = model.predict_proba(df)[0][1]

    return {"no_show_risk": round(float(prob), 3)}


@app.get("/debug")
def debug():
    return {
        "model_loaded": model is not None,
        "features_loaded": len(model_features),
        "first_10_features": model_features[:10]
    }
