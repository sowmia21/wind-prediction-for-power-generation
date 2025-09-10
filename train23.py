import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Mapping if your columns are named differently in other datasets
COLUMN_MAP = {
    'Temperature_2m': 'temperature_2m',
    'RelativeHumidity_2m': 'relativehumidity_2m',
    'DewPoint_2m': 'dewpoint_2m',
    'WindSpeed_10m': 'windspeed_10m',
    'WindSpeed_100m': 'windspeed_100m',
    'WindDirection_10m': 'winddirection_10m',
    'WindDirection_100m': 'winddirection_100m',
    'WindGusts_10m': 'windgusts_10m',
    'Power': 'Power'
}

def map_and_rename(df, column_map):
    df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})
    return df

def add_time_features(df, time_col):
    df[time_col] = pd.to_datetime(df[time_col])
    df['hour'] = df[time_col].dt.hour
    df['month'] = df[time_col].dt.month
    df['dayofweek'] = df[time_col].dt.dayofweek
    # cyclic encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek']/7)
    df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek']/7)
    return df

def add_lags(df, target_col, lags=[1,2,3]):
    for lag in lags:
        df[f'{target_col}_lag{lag}'] = df[target_col].shift(lag)
    return df

def preprocess(df):
    df = map_and_rename(df, COLUMN_MAP)

    # Detect target column automatically
    target_candidates = [c for c in df.columns if "power" in c.lower()]
    if not target_candidates:
        raise RuntimeError("No column containing 'power' found in the dataset!")
    target_col = target_candidates[0]

    # Add time features
    if "timestamp" in df.columns or "Time" in df.columns:
        ts_col = "timestamp" if "timestamp" in df.columns else "Time"
        df = add_time_features(df, ts_col)

    # Add lag features
    df = add_lags(df, target_col)

    # Drop rows with NaN from lags
    df = df.dropna().reset_index(drop=True)

    # Drop all non-numeric columns except target
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if target_col in non_numeric_cols:
        non_numeric_cols.remove(target_col)
    X = df.drop(columns=non_numeric_cols + [target_col], errors='ignore')

    y = df[target_col].values
    return X, y

def train(data_path, test_ratio=0.2, n_neighbors=5):
    print(f"Loading dataset: {data_path}")
    df = pd.read_csv(data_path)
    X, y = preprocess(df)

    print(f"Features used: {list(X.columns)}\n")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    print(f"Training KNN model with {n_neighbors} neighbors...\n")
    model = KNeighborsRegressor(n_neighbors=n_neighbors)
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)

    print("Sample Predictions vs Actual:")
    for i in range(min(5, len(y_test))):
        print(f"Predicted: {y_pred[i]:.2f} | Actual: {y_test[i]:.2f}")

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # <-- fixed for older sklearn
    r2 = r2_score(y_test, y_pred)
    accuracy = r2 * 100

    print("\nTraining complete.")
    print(f"Metrics: {{'MAE': {mae:.4f}, 'RMSE': {rmse:.4f}, 'R2': {r2:.4f}, 'Accuracy (%)': {accuracy:.2f}}}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="CSV file path")
    parser.add_argument("--test_ratio", type=float, default=0.2, help="Test data ratio")
    parser.add_argument("--neighbors", type=int, default=5, help="Number of neighbors for KNN")
    args = parser.parse_args()

    train(args.data, test_ratio=args.test_ratio, n_neighbors=args.neighbors)
