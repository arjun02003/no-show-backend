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
    try:
        # user input
        input_data = data.dict()

        # initialize ALL features = 0
        row = {feature: 0 for feature in model_features}

        # fill user provided values
        for k, v in input_data.items():
            if k in row:
                row[k] = int(v)

        # dataframe with exact order
        df = pd.DataFrame([row])
        df = df[model_features]

        # 🔴 CRITICAL FIX: force numeric dtype
        df = df.astype(float)

        prob = model.predict_proba(df)[0][1]

        return {
            "no_show_risk": round(float(prob), 3)
        }

    except Exception as e:
        # so we SEE the real error instead of 500
        return {
            "error": str(e)
        }



@app.get("/debug")
def debug():
    return {
        "model_loaded": True,
        "features_count": len(model_features),
        "first_10_features": model_features[:10]
    }
