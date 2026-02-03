from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# ---------------- LOAD MODEL ----------------
model = joblib.load("xgboost_no_show_model.pkl")
model_features = joblib.load("model_features.pkl")

# 🔥 CRITICAL FIX: attach feature names to booster
model.get_booster().feature_names = model_features

# ---------------- INPUT SCHEMA ----------------
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

# ---------------- ROUTES ----------------
@app.get("/")
def home():
    return {"status": "No-Show Prediction API is running"}

@app.post("/predict")
def predict(data: Patient):
    try:
        # initialize all features = 0
        row = {f: 0 for f in model_features}

        # fill user input
        for k, v in data.dict().items():
            if k in row:
                row[k] = int(v)

        # dataframe with exact training features
        df = pd.DataFrame([row])
        df = df[model_features]
        df = df.astype(float)

        prob = model.predict_proba(df)[0][1]

        return {
            "no_show_risk": round(float(prob), 3)
        }

    except Exception as e:
        return {"error": str(e)}

@app.get("/debug")
def debug():
    return {
        "model_loaded": True,
        "features_loaded": len(model_features),
        "first_10_features": model_features[:10]
    }

