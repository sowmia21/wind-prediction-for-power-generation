# train_model.py
import argparse
import os
import json
import pandas as pd
import numpy as np
import joblib
import math

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Models
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
import lightgbm as lgb
from catboost import CatBoostRegressor

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

def build_model(model_name, params):
    if model_name == "knn":
        return KNeighborsRegressor(
            n_neighbors=params.get("n_neighbors", 5),
            weights=params.get("weights", "uniform"),
            metric=params.get("metric", "minkowski"),
            p=params.get("p", 2)
        )
    elif model_name == "svr":
        return SVR(
            kernel=params.get("kernel", "rbf"),
            C=params.get("C", 1.0),
            epsilon=params.get("epsilon", 0.1)
        )
    elif model_name == "mlp":
        return MLPRegressor(
            hidden_layer_sizes=tuple(params.get("layers", [100])),
            max_iter=params.get("max_iter", 500),
            activation=params.get("activation", "relu"),
            solver=params.get("solver", "adam"),
            random_state=42
        )
    elif model_name == "xgboost":
        return XGBRegressor(
            n_estimators=params.get("n_estimators", 100),
            learning_rate=params.get("learning_rate", 0.1),
            max_depth=params.get("max_depth", 5),
            random_state=42,
            objective='reg:squarederror'
        )
    elif model_name == "lightgbm":
        return lgb.LGBMRegressor(
            n_estimators=params.get("n_estimators", 500),
            learning_rate=params.get("learning_rate", 0.05),
            max_depth=params.get("max_depth", -1),
            num_leaves=params.get("num_leaves", 31),
            subsample=params.get("subsample", 0.8),
            colsample_bytree=params.get("colsample_bytree", 0.8),
            random_state=42,
            n_jobs=-1
        )
    elif model_name == "catboost":
        return CatBoostRegressor(
            iterations=params.get("iterations", 500),
            learning_rate=params.get("learning_rate", 0.1),
            depth=params.get("depth", 6),
            loss_function='RMSE',
            verbose=100,
            random_seed=42
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def calculate_accuracy(y_true, y_pred):
    return 100 * (1 - (mean_absolute_error(y_true, y_pred) / np.mean(np.abs(y_true))))

def train(data_path, model_name="knn", test_ratio=0.2, scale=True, **kwargs):
    if not os.path.isfile(data_path):
        print(f"Error: File '{data_path}' does not exist.")
        return

    print(f"\nLoading dataset: {data_path}")
    df = pd.read_csv(data_path).dropna()
    target_column = df.columns[-1]
    X_df, y = preprocess(df, target_column)
    X = X_df.values

    print(f"Features used: {list(X_df.columns)}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=42
    )

    # Scaling if required
    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    # Model
    model = build_model(model_name.lower(), kwargs)
    print(f"\nTraining {model_name} model...")
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = math.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    acc = calculate_accuracy(y_test, y_pred)

    metrics = {"MAE": mae, "RMSE": rmse, "R2": r2, "Accuracy (%)": acc}
    
    print("\nSample Predictions vs Actual:")
    for p, a in list(zip(y_pred, y_test))[:5]:
        print(f"Predicted: {p:.2f} | Actual: {a:.2f}")

    print("\nTraining complete.")
    print("Metrics:", metrics)

    # Save artifacts
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(model, os.path.join("artifacts", "model.pkl"))
    if scaler:
        joblib.dump(scaler, os.path.join("artifacts", "scaler.pkl"))
    with open(os.path.join("artifacts", "features.json"), "w") as f:
        json.dump(list(X_df.columns), f, indent=2)
    with open(os.path.join("artifacts", "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return model, scaler, metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train any regression model on Wind Power Data")
    parser.add_argument("--data", required=True, help="Path to CSV dataset")
    parser.add_argument("--model", default="knn", help="Model: knn, svr, mlp, xgboost, lightgbm, catboost")
    parser.add_argument("--test_ratio", type=float, default=0.2, help="Test set fraction")
    parser.add_argument("--scale", type=bool, default=True, help="Apply StandardScaler")
    parser.add_argument("--params", type=json.loads, default="{}", help="JSON string of model hyperparameters")

    args = parser.parse_args()
    train(args.data, model_name=args.model, test_ratio=args.test_ratio, scale=args.scale, **args.params)
