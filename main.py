# main.py
from fastapi import FastAPI
from pydantic import BaseModel, Field
from datetime import datetime
import pandas as pd
import numpy as np
import joblib
import json, os

# Paths to artifacts and config
ARTIFACT_DIR = "artifacts"
FEATURES_PATH = os.path.join(ARTIFACT_DIR, "features.json")
MODEL_PATH = os.path.join(ARTIFACT_DIR, "model.pkl")
SCALER_PATH = os.path.join(ARTIFACT_DIR, "scaler.pkl")
CONFIG_PATH = "config.json"

PRED_CSV = "predictions_log.csv"
PRED_JSON = "predictions_log.json"

app = FastAPI(title="Wind Power Prediction (Linear Regression)")

# Define request schema
class PredictRequest(BaseModel):
    temperature_2m: float = Field(None, description="Air temperature at 2m")
    relativehumidity_2m: float = Field(None, description="Relative humidity at 2m")
    dewpoint_2m: float = Field(None, description="Dew point at 2m")
    windspeed_10m: float = Field(..., description="Wind speed at 10m (m/s)")
    windspeed_100m: float = Field(None, description="Wind speed at 100m (m/s)")
    winddirection_10m: float = Field(None, description="Wind direction at 10m (deg)")
    winddirection_100m: float = Field(None, description="Wind direction at 100m (deg)")
    windgusts_10m: float = Field(None, description="Wind gusts at 10m (m/s)")
    timestamp: str = Field(None, description="ISO timestamp (optional)")

# Lazy-load model, scaler, features, config
def load_runtime():
    model = scaler = features = None
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(FEATURES_PATH):
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        with open(FEATURES_PATH, "r") as f:
            features = json.load(f)
    cfg = {}
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            cfg = json.load(f)
    return model, scaler, features, cfg

# Physics-based fallback
def physics_power_kw(v, cfg):
    pc = cfg.get("power_curve", {}) if cfg else {}
    rho = pc.get("rho", 1.225)
    d = pc.get("rotor_diameter_m", 100.0)
    cp = pc.get("cp", 0.42)
    eff = pc.get("efficiency", 0.9)
    cut_in = pc.get("cut_in", 3.0)
    rated = pc.get("rated", 12.0)
    cut_out = pc.get("cut_out", 25.0)
    rated_kw = pc.get("rated_power_kw", 2500.0)
    if v < cut_in or v >= cut_out:
        return 0.0
    if v >= rated:
        return float(rated_kw)
    area = np.pi * (d/2)**2
    p_watts = 0.5 * rho * area * cp * (v ** 3) * eff
    return float(p_watts / 1000.0)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    model, scaler, features, cfg = load_runtime()

    # Build input row
    row = {}

    # Time features
    if req.timestamp:
        ts = pd.to_datetime(req.timestamp, errors="coerce")
        hour = int(ts.hour) if not pd.isna(ts) else 0
        month = int(ts.month) if not pd.isna(ts) else 1
        row["hour"] = hour
        row["month"] = month
        row["hour_sin"] = float(np.sin(2 * np.pi * hour / 24))
        row["hour_cos"] = float(np.cos(2 * np.pi * hour / 24))
        row["month_sin"] = float(np.sin(2 * np.pi * month / 12))
        row["month_cos"] = float(np.cos(2 * np.pi * month / 12))

    # Sensor inputs
    row["windspeed_10m"] = req.windspeed_10m
    if req.windspeed_100m is not None: row["windspeed_100m"] = req.windspeed_100m
    if req.temperature_2m is not None: row["temperature_2m"] = req.temperature_2m
    if req.relativehumidity_2m is not None: row["relativehumidity_2m"] = req.relativehumidity_2m
    if req.dewpoint_2m is not None: row["dewpoint_2m"] = req.dewpoint_2m
    if req.winddirection_10m is not None: row["winddirection_10m"] = req.winddirection_10m
    if req.winddirection_100m is not None: row["winddirection_100m"] = req.winddirection_100m
    if req.windgusts_10m is not None: row["windgusts_10m"] = req.windgusts_10m

    # If model not loaded, return physics fallback
    if model is None or scaler is None or features is None:
        p_kw = physics_power_kw(req.windspeed_10m, cfg)
        res = {
            "prediction_timestamp": datetime.now().isoformat(),
            "inputs": {
                "windspeed_10m": req.windspeed_10m,
                "temperature_2m": req.temperature_2m,
                "relativehumidity_2m": req.relativehumidity_2m,
                "windgusts_10m": req.windgusts_10m,
                "timestamp": req.timestamp
            },
            "predicted_generation": {"power_kw": float(p_kw), "power_mw": float(p_kw/1000.0)},
            "status": "physics_fallback"
        }
        _append_logs(res)
        return res

    # Align features with training
    for f in features:
        if f not in row:
            row[f] = 0.0

    X = pd.DataFrame([row])[features]
    Xs = scaler.transform(X.values)
    pred = model.predict(Xs)[0]

    res = {
        "prediction_timestamp": datetime.now().isoformat(),
        "inputs": {
            "windspeed_10m": req.windspeed_10m,
            "temperature_2m": req.temperature_2m,
            "relativehumidity_2m": req.relativehumidity_2m,
            "windgusts_10m": req.windgusts_10m,
            "timestamp": req.timestamp
        },
        "predicted_generation": {"power_kw": float(pred), "power_mw": float(pred/1000.0)},
        "status": "forecasted"
    }
    _append_logs(res)
    return res

def _append_logs(res):
    """
    Append predictions to CSV and JSON logs
    """
    row = {
        "timestamp": res["prediction_timestamp"],
        "windspeed_10m": res["inputs"].get("windspeed_10m"),
        "temperature_2m": res["inputs"].get("temperature_2m"),
        "relativehumidity_2m": res["inputs"].get("relativehumidity_2m"),
        "power_kw": res["predicted_generation"]["power_kw"],
        "power_mw": res["predicted_generation"]["power_mw"],
        "status": res["status"]
    }
    df = pd.DataFrame([row])
    if not os.path.exists(PRED_CSV):
        df.to_csv(PRED_CSV, index=False)
    else:
        df.to_csv(PRED_CSV, mode="a", header=False, index=False)

    if not os.path.exists(PRED_JSON):
        with open(PRED_JSON, "w") as f:
            json.dump([res], f, indent=2)
    else:
        with open(PRED_JSON, "r+") as f:
            data = json.load(f)
            data.append(res)
            f.seek(0)
            json.dump(data, f, indent=2)
