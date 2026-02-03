from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load("xgboost_no_show_model.pkl")

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

@app.post("/predict")
def predict(data: Patient):
    input_dict = data.dict()

    df = pd.DataFrame([input_dict])

    # Load feature list manually from training
    expected_features = model.get_booster().feature_names

    for col in expected_features:
        if col not in df.columns:
            df[col] = 0

    df = df[expected_features]

    prob = model.predict_proba(df)[0][1]
    return {"no_show_risk": round(float(prob), 3)}



