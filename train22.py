# train21.py
import argparse
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def preprocess(df, target="Power"):
    # Convert datetime columns to numeric features
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_datetime(df[col])
                df[col + '_hour'] = df[col].dt.hour
                df[col + '_day'] = df[col].dt.day
                df[col + '_month'] = df[col].dt.month
                df[col + '_weekday'] = df[col].dt.weekday
                # Add sin/cos encoding for cyclic features
                df[col + '_hour_sin'] = np.sin(2 * np.pi * df[col + '_hour'] / 24)
                df[col + '_hour_cos'] = np.cos(2 * np.pi * df[col + '_hour'] / 24)
                df[col + '_month_sin'] = np.sin(2 * np.pi * df[col + '_month'] / 12)
                df[col + '_month_cos'] = np.cos(2 * np.pi * df[col + '_month'] / 12)
                df[col + '_weekday_sin'] = np.sin(2 * np.pi * df[col + '_weekday'] / 7)
                df[col + '_weekday_cos'] = np.cos(2 * np.pi * df[col + '_weekday'] / 7)
                df.drop(columns=[col], inplace=True)
            except Exception:
                df.drop(columns=[col], inplace=True)
    
    # Add lag features for target
    if target in df.columns:
        for lag in range(1, 4):  # lag1, lag2, lag3
            df[f"{target}_lag{lag}"] = df[target].shift(lag)
    df = df.dropna()  # drop rows with NaNs after lag

    X = df.drop(columns=[target])
    y = df[target].values
    return X, y

def train(data_path, test_ratio=0.2, n_neighbors=5):
    if not os.path.isfile(data_path):
        print(f"Error: File '{data_path}' does not exist.")
        return

    print(f"Loading dataset: {data_path}")
    df = pd.read_csv(data_path)

    X_df, y = preprocess(df, target="Power")
    X = X_df.values

    print(f"Features used: {list(X_df.columns)}")

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=42, shuffle=False  # time-series aware split
    )

    # Scaling features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # --- KNN Regressor ---
    model = KNeighborsRegressor(
        n_neighbors=n_neighbors,
        weights='distance',  # distance-weighted neighbors
        metric='minkowski',
        p=2
    )

    print(f"\nTraining KNN model with {n_neighbors} neighbors...")
    model.fit(X_train_s, y_train)

    # --- Evaluation ---
    y_pred = model.predict(X_test_s)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)
    accuracy = max(0, min(100, r2 * 100))  # convert R2 to percentage

    print("\nSample Predictions vs Actual:")
    for i in range(min(5, len(y_test))):
        print(f"Predicted: {y_pred[i]:.2f} | Actual: {y_test[i]:.2f}")

    print("\nTraining complete.")
    print(f"Metrics: {{'MAE': {mae:.4f}, 'RMSE': {rmse:.4f}, 'R2': {r2:.4f}, 'Accuracy (%)': {accuracy:.2f}}}")

    return model, scaler

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train KNN Regressor on Wind Power Data (Enhanced)")
    parser.add_argument('--data', type=str, required=True, help="Path to CSV dataset")
    parser.add_argument('--test_ratio', type=float, default=0.2, help="Test set ratio")
    parser.add_argument('--neighbors', type=int, default=5, help="Number of neighbors for KNN")
    args = parser.parse_args()

    data_path = args.data.strip('"').strip("'")
    train(data_path, test_ratio=args.test_ratio, n_neighbors=args.neighbors)
