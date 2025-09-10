# train21.py
import argparse
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# -------------------
# Feature preprocessing
# -------------------
def add_time_features(df, time_col="Time"):
    """Convert datetime column to numeric features"""
    if time_col in df.columns:
        try:
            ts = pd.to_datetime(df[time_col], errors="coerce")
            df[time_col + "_hour"] = ts.dt.hour
            df[time_col + "_day"] = ts.dt.day
            df[time_col + "_month"] = ts.dt.month
            df[time_col + "_weekday"] = ts.dt.weekday
            df.drop(columns=[time_col], inplace=True)
        except Exception:
            df.drop(columns=[time_col], inplace=True)
    return df

def add_lags_rolls(df, col, lags=[1,3,6], rolls=[3,6]):
    """Add lag and rolling mean features"""
    if col in df.columns:
        for l in lags:
            df[f"{col}_lag{l}"] = df[col].shift(l)
        for w in rolls:
            df[f"{col}_roll{w}"] = df[col].rolling(window=w, min_periods=1).mean()
    return df

def preprocess(df, target="Power"):
    # Add time features
    df = add_time_features(df, "Time")
    # Add lags and rolling features for windspeed
    df = add_lags_rolls(df, "windspeed_10m")
    df = add_lags_rolls(df, "windspeed_100m")
    df = add_lags_rolls(df, "power") if "power" in df.columns else df

    # Features: only numeric
    X = df.drop(columns=[target])
    X = X.select_dtypes(include=[np.number])
    
    y = df[target].values

    # Remove rows with NaN after lagging
    valid_idx = ~np.isnan(X).any(axis=1)
    X = X.loc[valid_idx].reset_index(drop=True)
    y = y[valid_idx]

    return X, y

# -------------------
# Train function
# -------------------
def train(data_path, test_ratio=0.2, n_neighbors=5):
    if not os.path.isfile(data_path):
        print(f"Error: File '{data_path}' does not exist.")
        return

    print(f"Loading dataset: {data_path}")
    df = pd.read_csv(data_path)
    df = df.dropna(subset=["Power"])  # remove rows where target is NaN

    X, y = preprocess(df, target="Power")
    print(f"Features used: {list(X.columns)}")

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y, test_size=test_ratio, random_state=42
    )

    # Scale features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Train KNN
    model = KNeighborsRegressor(n_neighbors=n_neighbors)
    print(f"\nTraining knn model with {n_neighbors} neighbors...")
    model.fit(X_train_s, y_train)

    # Predictions
    y_pred = model.predict(X_test_s)

    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)
    # Accuracy (%) = R2 × 100
    accuracy = r2 * 100

    print("\nSample Predictions vs Actual:")
    for i in range(min(5, len(y_test))):
        print(f"Predicted: {y_pred[i]:.2f} | Actual: {y_test[i]:.2f}")

    print("\nTraining complete.")
    print(f"Metrics: {{'MAE': {mae:.4f}, 'RMSE': {rmse:.4f}, 'R2': {r2:.4f}, 'Accuracy (%)': {accuracy:.2f}}}")

    return model, scaler

# -------------------
# CLI
# -------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train KNN on Wind Power Data")
    parser.add_argument('--data', type=str, required=True, help="Path to CSV dataset")
    parser.add_argument('--test_ratio', type=float, default=0.2, help="Test set fraction")
    parser.add_argument('--neighbors', type=int, default=5, help="Number of neighbors for KNN")
    args = parser.parse_args()

    data_path = args.data.strip('"').strip("'")
    train(data_path, test_ratio=args.test_ratio, n_neighbors=args.neighbors)


