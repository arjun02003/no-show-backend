from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# Load model + features
model = joblib.load("xgboost_no_show_model.pkl")
model_features = joblib.load("model_features.pkl")

# ---------- INPUT SCHEMA ----------
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

    Gender: str              # "M" or "F"
    Neighbourhood: str       # e.g. "JARDIM CAMBURI"
    Season: str              # e.g. "Summer"


@app.get("/")
def home():
    return {"status": "No-Show Prediction API running"}


@app.post("/predict")
def predict(data: Patient):

    # 1️⃣ Empty dataframe with ALL model features
    df = pd.DataFrame(0, index=[0], columns=model_features)

    # 2️⃣ Fill numeric features
    numeric_fields = [
        "Age", "LeadTimeDays", "ScheduledHour", "AppointmentDayOfWeek",
        "SMS_received", "PastNoShows", "Scholarship",
        "Hipertension", "Diabetes", "Alcoholism", "Handcap"
    ]

    for f in numeric_fields:
        df.at[0, f] = getattr(data, f)

    # 3️⃣ Gender one-hot
    gender_col = f"Gender_{data.Gender.upper()}"
    if gender_col in df.columns:
        df.at[0, gender_col] = 1

    # 4️⃣ Neighbourhood one-hot
    neigh_col = f"Neighbourhood_{data.Neighbourhood.upper()}"
    if neigh_col in df.columns:
        df.at[0, neigh_col] = 1

    # 5️⃣ Season one-hot
    season_col = f"Season_{data.Season}"
    if season_col in df.columns:
        df.at[0, season_col] = 1

    # 6️⃣ Predict
    prob = model.predict_proba(df)[0][1]

    return {
        "no_show_risk": round(float(prob), 3)
    }


@app.get("/debug")
def debug():
    return {
        "total_features": len(model_features),
        "first_10": model_features[:10]
    }
