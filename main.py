from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import pandas as pd
import sqlite3
from datetime import datetime

app = FastAPI()

# 🔥 CORS (Frontend ke liye required)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # local + deploy dono ke liye ok
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- LOAD MODEL ----------
model = joblib.load("xgboost_no_show_model.pkl")
model_features = joblib.load("model_features.pkl")

# ---------- DATABASE SETUP ----------
conn = sqlite3.connect("predictions.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    age INTEGER,
    lead_time INTEGER,
    scheduled_hour INTEGER,
    appointment_day INTEGER,
    gender TEXT,
    neighbourhood TEXT,
    season TEXT,
    risk REAL,
    created_at TEXT
)
""")
conn.commit()

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
    Gender: str
    Neighbourhood: str
    Season: str

# ---------- ROUTES ----------
@app.get("/")
def home():
    return {"status": "No-Show Prediction API running"}

@app.get("/ui")
def ui():
    return FileResponse("index.html")

@app.post("/predict")
def predict(data: Patient):

    # ----- Build input dataframe -----
    df = pd.DataFrame(0, index=[0], columns=model_features)

    numeric_fields = [
        "Age", "LeadTimeDays", "ScheduledHour", "AppointmentDayOfWeek",
        "SMS_received", "PastNoShows", "Scholarship",
        "Hipertension", "Diabetes", "Alcoholism", "Handcap"
    ]

    for f in numeric_fields:
        df.at[0, f] = getattr(data, f)

    # ----- One-hot encoding -----
    gcol = f"Gender_{data.Gender.upper()}"
    if gcol in df.columns:
        df.at[0, gcol] = 1

    ncol = f"Neighbourhood_{data.Neighbourhood.upper()}"
    if ncol in df.columns:
        df.at[0, ncol] = 1

    scol = f"Season_{data.Season}"
    if scol in df.columns:
        df.at[0, scol] = 1

    # ----- Prediction -----
    prob = model.predict_proba(df)[0][1]
    risk_value = round(float(prob), 3)

    # ----- SAVE TO DATABASE -----
    cursor.execute("""
        INSERT INTO predictions
        (age, lead_time, scheduled_hour, appointment_day,
         gender, neighbourhood, season, risk, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        data.Age,
        data.LeadTimeDays,
        data.ScheduledHour,
        data.AppointmentDayOfWeek,
        data.Gender,
        data.Neighbourhood,
        data.Season,
        risk_value,
        datetime.now().isoformat()
    ))
    conn.commit()

    return {"no_show_risk": risk_value}

# ---------- VIEW SAVED DATA (OPTIONAL BUT COOL) ----------
@app.get("/history")
def history():
    cursor.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT 10")
    rows = cursor.fetchall()
    return rows

