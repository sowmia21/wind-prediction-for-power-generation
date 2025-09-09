# train_xgboost.py
import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

def preprocess(df, target_column):
    # Convert datetime columns to numeric features
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_datetime(df[col])
                df[col + '_hour'] = df[col].dt.hour
                df[col + '_day'] = df[col].dt.day
                df[col + '_month'] = df[col].dt.month
                df[col + '_weekday'] = df[col].dt.weekday
                df.drop(columns=[col], inplace=True)
            except Exception:
                df.drop(columns=[col], inplace=True)
    
    X = df.drop(columns=[target_column])
    y = df[target_column].values
    return X, y

def train(data_path, test_ratio=0.2):
    if not os.path.isfile(data_path):
        print(f"Error: File '{data_path}' does not exist.")
        return

    print(f"Loading dataset: {data_path}")
    df = pd.read_csv(data_path)
    df = df.dropna()
    
    target_column = df.columns[-1]
    X_df, y = preprocess(df, target_column)
    X = X_df.values

    print(f"Features used for training: {list(X_df.columns)}")

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=42
    )

    # Optional scaling (XGBoost usually handles raw features fine)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # --- XGBoost Regressor ---
    model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        objective='reg:squarederror'
    )

    print("Training XGBoost Regressor...")
    model.fit(X_train_s, y_train)

    # --- Evaluation ---
    y_pred = model.predict(X_test_s)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)

    print("\nSample Predictions vs Actual Power:")
    for i in range(min(5, len(y_test))):
        print(f"Predicted: {y_pred[i]:.2f} kW | Actual: {y_test[i]:.2f} kW")

    print("\nTraining complete.")
    print(f"Metrics: {{'MAE': {mae:.4f}, 'RMSE': {rmse:.4f}, 'R2': {r2:.4f}}}")

    return model, scaler

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XGBoost Regressor on Wind Power Data")
    parser.add_argument('--data', type=str, required=True, help="Path to CSV dataset")
    parser.add_argument('--test_ratio', type=float, default=0.2, help="Test set ratio")
    args = parser.parse_args()

    data_path = args.data.strip('"').strip("'")
    train(data_path, test_ratio=args.test_ratio)
