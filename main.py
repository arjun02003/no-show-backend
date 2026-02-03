from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd


app = FastAPI()

# load model + feature list
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
    Handcap: int


@app.get("/")
def home():
    return {"status": "No-Show Prediction API is running"}

@app.post("/predict")
def predict(data: Patient):
    # user input
    input_dict = data.dict()

    # create dataframe
    df = pd.DataFrame([input_dict])

    # ensure ALL model features exist
    for col in model_features:
        if col not in df.columns:
            df[col] = 0

    # reorder exactly as training
    df = df[model_features]

    # prediction
    prob = model.predict_proba(df)[0][1]

    return {
        "no_show_risk": round(float(prob), 3)
    }



@app.get("/debug")
def debug():
    return {
        "model_loaded": True,
        "features_loaded": len(model_features),
        "first_10_features": model_features[:10]
    }

@app.get("/debug")
def debug():
    return {
        "model_loaded": True,
        "features_loaded": len(model_features),
        "first_10_features": model_features[:10]
    }


