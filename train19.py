# train_ensemble_fixed.py
import argparse
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

def preprocess(df, target_column):
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

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # Compatible fix
    r2 = r2_score(y_test, y_pred)
    return mae, rmse, r2, y_pred

def train_models(data_path, test_ratio=0.2, n_neighbors=5, rf_estimators=100, xgb_rounds=500):
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

    # Scale features for KNN
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # --- KNN ---
    knn = KNeighborsRegressor(n_neighbors=n_neighbors, weights='distance')
    knn.fit(X_train_s, y_train)
    knn_mae, knn_rmse, knn_r2, knn_pred = evaluate_model(knn, X_test_s, y_test)
    print(f"\nKNN Metrics: MAE={knn_mae:.4f}, RMSE={knn_rmse:.4f}, R2={knn_r2:.4f}")

    # --- Random Forest ---
    rf = RandomForestRegressor(n_estimators=rf_estimators, random_state=42)
    rf.fit(X_train, y_train)
    rf_mae, rf_rmse, rf_r2, rf_pred = evaluate_model(rf, X_test, y_test)
    print(f"\nRandom Forest Metrics: MAE={rf_mae:.4f}, RMSE={rf_rmse:.4f}, R2={rf_r2:.4f}")

    # --- XGBoost ---
    xgbr = xgb.XGBRegressor(n_estimators=xgb_rounds, learning_rate=0.05, random_state=42)
    xgbr.fit(X_train, y_train)
    xgb_mae, xgb_rmse, xgb_r2, xgb_pred = evaluate_model(xgbr, X_test, y_test)
    print(f"\nXGBoost Metrics: MAE={xgb_mae:.4f}, RMSE={xgb_rmse:.4f}, R2={xgb_r2:.4f}")

    # --- Ensemble (average) ---
    ensemble_pred = (knn_pred + rf_pred + xgb_pred) / 3
    ens_mae = mean_absolute_error(y_test, ensemble_pred)
    ens_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
    ens_r2 = r2_score(y_test, ensemble_pred)
    print(f"\nEnsemble Metrics: MAE={ens_mae:.4f}, RMSE={ens_rmse:.4f}, R2={ens_r2:.4f}")

    # --- Sample predictions ---
    print("\nSample Predictions (Ensemble vs Actual):")
    for i in range(min(5, len(y_test))):
        print(f"Predicted: {ensemble_pred[i]:.2f} | Actual: {y_test[i]:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train KNN, Random Forest, XGBoost on Wind Data")
    parser.add_argument('--data', type=str, required=True, help="Path to CSV dataset")
    parser.add_argument('--test_ratio', type=float, default=0.2, help="Test set ratio")
    parser.add_argument('--neighbors', type=int, default=5, help="K for KNN")
    parser.add_argument('--rf_estimators', type=int, default=100, help="Random Forest trees")
    parser.add_argument('--xgb_rounds', type=int, default=500, help="XGBoost rounds")
    args = parser.parse_args()

    train_models(
        data_path=args.data.strip('"').strip("'"),
        test_ratio=args.test_ratio,
        n_neighbors=args.neighbors,
        rf_estimators=args.rf_estimators,
        xgb_rounds=args.xgb_rounds
    )
