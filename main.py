from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import traceback

app = FastAPI(title="No-Show Prediction API")

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
    return {"status": "No-Show Prediction API running"}


@app.post("/predict")
def predict(data: Patient):
    try:
        # 1. Input dataframe
        df = pd.DataFrame([data.dict()])

        # 2. Add missing features
        for col in model_features:
            if col not in df.columns:
                df[col] = 0

        # 3. Reorder
        df = df[model_features]

        # 4. Force numeric (CRITICAL FIX)
        df = df.apply(pd.to_numeric)

        # 5. Predict
        prob = model.predict_proba(df)[0][1]

        return {"no_show_risk": round(float(prob), 3)}

    except Exception as e:
        # PRINT FULL ERROR TO RENDER LOGS
        print("❌ PREDICTION ERROR")
        print(traceback.format_exc())
        return {"error": str(e)}


@app.get("/debug")
def debug():
    return {
        "model_loaded": model is not None,
        "features_count": len(model_features),
        "first_10_features": model_features[:10]
    }
