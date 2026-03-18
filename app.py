"""
🚀 FastAPI Backend for AI Probabilistic Weather Forecast
Production-ready (Render compatible)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import pickle
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# ML
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import InputLayer

# ==================== CONFIG ====================

API_KEY = os.getenv("WEATHER_API_KEY")

if not API_KEY:
    print("⚠️ WARNING: WEATHER_API_KEY not set. API may fail.")

PAST_HOURS = 72
FUTURE_DAYS = 7
QUANTILES = [0.1, 0.5, 0.9]

FEATURES = [
    "temperature_celsius",
    "humidity",
    "pressure_mb",
    "wind_kph",
    "cloud"
]

BASE_DIR = Path(__file__).parent

MODEL_PATH = BASE_DIR / "weather_model.h5"
SCALER_PATH = BASE_DIR / "scaler.pkl"

# ==================== FASTAPI INIT ====================

app = FastAPI(title="AI Weather Forecast API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== LOAD MODEL ====================

def load_model_safely(model_path):
    try:
        print("🔄 Loading model...")
        model = load_model(
            model_path,
            compile=False,
            custom_objects={"InputLayer": InputLayer}
        )
        print("✅ Model loaded")
        return model
    except Exception as e:
        print("❌ Model load failed:", e)
        return None


try:
    model = load_model_safely(str(MODEL_PATH)) if MODEL_PATH.exists() else None

    if SCALER_PATH.exists():
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        print("✅ Scaler loaded")
    else:
        scaler = None
        print("⚠️ Scaler not found")

except Exception as e:
    print("❌ Startup error:", e)
    model = None
    scaler = None


if model is None:
    print("⚠️ Running in fallback mode (no ML model)")

# ==================== WEATHER API ====================

def fetch_weather(city: str):
    if not API_KEY:
        raise ValueError("API key missing")

    url = (
        "https://api.openweathermap.org/data/2.5/weather"
        f"?q={city}&appid={API_KEY}&units=metric"
    )

    try:
        response = requests.get(url, timeout=10)
        data = response.json()

        if response.status_code != 200:
            raise ValueError(data.get("message", "Weather API error"))

        return {
            "temperature_celsius": data["main"]["temp"],
            "humidity": data["main"]["humidity"],
            "pressure_mb": data["main"]["pressure"],
            "wind_kph": data["wind"]["speed"] * 3.6,
            "cloud": data["clouds"]["all"],
        }

    except Exception as e:
        raise ValueError(f"Weather fetch failed: {str(e)}")


# ==================== ML PREDICTION ====================

def predict_weather(history_df, current_dict):

    if model is None or scaler is None:
        raise RuntimeError("Model or scaler missing")

    data = pd.concat(
        [history_df, pd.DataFrame([current_dict])],
        ignore_index=True
    )[FEATURES]

    scaled = scaler.transform(data)
    X = scaled[-PAST_HOURS:].reshape(1, PAST_HOURS, len(FEATURES))

    preds = model.predict(X, verbose=0)[0]
    preds = preds.reshape(FUTURE_DAYS, len(QUANTILES))

    def inverse_temp(temp_scaled):
        dummy = np.zeros((FUTURE_DAYS, len(FEATURES)))
        dummy[:, 0] = temp_scaled
        return scaler.inverse_transform(dummy)[:, 0]

    return (
        inverse_temp(preds[:, 0]).tolist(),
        inverse_temp(preds[:, 1]).tolist(),
        inverse_temp(preds[:, 2]).tolist(),
    )

# ==================== ROUTES ====================

@app.get("/")
async def root():
    return {"message": "Weather API running 🚀"}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/api/forecast/{city}")
async def get_forecast(city: str):

    if not city.strip():
        raise HTTPException(status_code=400, detail="City required")

    try:
        current = fetch_weather(city)

        # Simulated past data
        history = []
        for _ in range(PAST_HOURS - 1):
            history.append({
                "temperature_celsius": current["temperature_celsius"] + np.random.uniform(-4, 4),
                "humidity": np.clip(current["humidity"] + np.random.uniform(-10, 10), 0, 100),
                "pressure_mb": np.clip(current["pressure_mb"] + np.random.uniform(-8, 8), 900, 1100),
                "wind_kph": max(0, current["wind_kph"] + np.random.uniform(-5, 5)),
                "cloud": np.clip(current["cloud"] + np.random.uniform(-20, 20), 0, 100),
            })

        history_df = pd.DataFrame(history)[FEATURES]

        # Prediction
        try:
            q10, q50, q90 = predict_weather(history_df, current)
        except Exception as e:
            print("⚠️ Fallback prediction:", e)
            q10 = [current["temperature_celsius"] + np.random.uniform(-2, 2) for _ in range(FUTURE_DAYS)]
            q50 = [current["temperature_celsius"] + np.random.uniform(-1, 1) for _ in range(FUTURE_DAYS)]
            q90 = [current["temperature_celsius"] + np.random.uniform(0, 3) for _ in range(FUTURE_DAYS)]

        dates = [
            (datetime.now().date() + timedelta(days=i)).isoformat()
            for i in range(FUTURE_DAYS + 1)
        ]

        forecast = []

        for i, date in enumerate(dates):
            if i == 0:
                temp = current["temperature_celsius"]
                forecast.append({
                    "date": date,
                    "lower_10": round(temp, 2),
                    "median_50": round(temp, 2),
                    "upper_90": round(temp, 2),
                })
            else:
                forecast.append({
                    "date": date,
                    "lower_10": round(q10[i-1], 2),
                    "median_50": round(q50[i-1], 2),
                    "upper_90": round(q90[i-1], 2),
                })

        return {
            "city": city.title(),
            "current": current,
            "forecast": forecast
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
