# train2.py
import argparse
import os
import json
import math
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# ------- COLUMN MAP -------
COLUMN_MAP = {
    "timestamp": "Time",
    "temperature_2m": "temperature_2m",
    "relativehumidity_2m": "relativehumidity_2m",
    "dewpoint_2m": "dewpoint_2m",
    "windspeed_10m": "windspeed_10m",
    "windspeed_100m": "windspeed_100m",
    "winddirection_10m": "winddirection_10m",
    "winddirection_100m": "winddirection_100m",
    "windgusts_10m": "windgusts_10m",
    "power": "Power"
}
# --------------------------

def map_and_rename(df, colmap):
    rename_map = {v: k for k, v in colmap.items() if v in df.columns}
    return df.rename(columns=rename_map).copy()

def add_time_features(df, ts_col="timestamp"):
    ts = pd.to_datetime(df[ts_col], errors="coerce")
    df["hour"] = ts.dt.hour.fillna(0).astype(int)
    df["month"] = ts.dt.month.fillna(1).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df

def add_lags_and_rolls(df, lags=[1,3,6], rolls=[3,6]):
    if "windspeed_10m" in df.columns:
        for l in lags:
            df[f"windspeed_10m_lag{l}"] = df["windspeed_10m"].shift(l)
        for w in rolls:
            df[f"windspeed_10m_roll{w}"] = df["windspeed_10m"].rolling(window=w, min_periods=1).mean()
    return df

def build_features(df, cfg_map):
    df = map_and_rename(df, cfg_map)
    if cfg_map.get("timestamp", "") in df.columns:
        df = add_time_features(df, "timestamp")
    df = add_lags_and_rolls(df)

    feature_candidates = [
        "temperature_2m","relativehumidity_2m","dewpoint_2m",
        "windspeed_10m","windspeed_100m",
        "winddirection_10m","winddirection_100m","windgusts_10m",
        "hour","month","hour_sin","hour_cos","month_sin","month_cos"
    ]
    features = [c for c in feature_candidates if c in df.columns]
    features += [c for c in df.columns if ("_lag" in c) or ("_roll" in c)]

    X = df[features].copy()
    y = df["power"] if "power" in df.columns else None
    valid = ~X.isna().any(axis=1)
    X = X.loc[valid].reset_index(drop=True)
    y = y.loc[valid].reset_index(drop=True) if y is not None else None
    return X, y, features

def train(csv_path, test_ratio=0.2):
    print("Loading dataset:", csv_path)
    df = pd.read_csv(csv_path)
    df = map_and_rename(df, COLUMN_MAP)

    X, y, feat_names = build_features(df, COLUMN_MAP)
    if y is None:
        raise RuntimeError("Target 'Power' not found after mapping. Update COLUMN_MAP if needed.")

    n = len(X)
    split = int(n * (1 - test_ratio))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # ---- Random Forest Model ----
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_s, y_train)

    # ---- Evaluation ----
    preds = model.predict(X_test_s)
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = math.sqrt(mse)   # fix for sklearn issue
    r2 = r2_score(y_test, preds)

    metrics = {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2)
    }

    # --- Print sample predictions ---
    print("\nSample Predictions vs Actual Power:")
    for p, a in list(zip(preds, y_test))[:5]:
        print(f"Predicted: {p:.2f} kW | Actual: {a:.2f} kW")

    # ---- Save Artifacts ----
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(model, os.path.join("artifacts", "model.pkl"))
    joblib.dump(scaler, os.path.join("artifacts", "scaler.pkl"))
    with open(os.path.join("artifacts", "features.json"), "w") as f:
        json.dump(feat_names, f, indent=2)
    with open(os.path.join("artifacts", "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("Training complete. Artifacts saved to ./artifacts/")
    print("Metrics:", metrics)
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Random Forest on wind dataset")
    parser.add_argument("--data", required=True, help="Path to CSV dataset")
    parser.add_argument("--test_ratio", type=float, default=0.2, help="Test set fraction")
    args = parser.parse_args()
    train(args.data, test_ratio=args.test_ratio)
