import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import argparse

def load_and_combine(files):
    dfs = [pd.read_csv(f) for f in files]
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df

def add_time_features(df):
    df['hour'] = pd.to_datetime(df['Time']).dt.hour
    df['month'] = pd.to_datetime(df['Time']).dt.month
    df['dayofweek'] = pd.to_datetime(df['Time']).dt.dayofweek
    
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek']/7)
    df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek']/7)
    
    df['Power_lag1'] = df['Power'].shift(1)
    df['Power_lag2'] = df['Power'].shift(2)
    df['Power_lag3'] = df['Power'].shift(3)
    
    df = df.dropna().reset_index(drop=True)
    return df

def train_knn(df, n_neighbors=5, test_ratio=0.2):
    features = ['temperature_2m', 'relativehumidity_2m', 'dewpoint_2m',
                'windspeed_10m', 'windspeed_100m', 'winddirection_10m',
                'winddirection_100m', 'windgusts_10m', 'hour', 'month',
                'dayofweek', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
                'dow_sin', 'dow_cos', 'Power_lag1', 'Power_lag2', 'Power_lag3']
    
    X = df[features]
    y = df['Power']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_ratio, random_state=42)
    
    knn = KNeighborsRegressor(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    
    y_pred = knn.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # FIX: compute manually
    r2 = r2_score(y_test, y_pred)
    accuracy = r2 * 100
    
    print("\nSample Predictions vs Actual:")
    for pred, actual in zip(y_pred[:5], y_test[:5]):
        print(f"Predicted: {pred:.2f} | Actual: {actual:.2f}")
    
    print("\nTraining complete.")
    print(f"Metrics: {{'MAE': {mae:.4f}, 'RMSE': {rmse:.4f}, 'R2': {r2:.4f}, 'Accuracy (%)': {accuracy:.2f}}}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--files', nargs='+', required=True, help="List of CSV files to combine")
    parser.add_argument('--neighbors', type=int, default=5, help="Number of KNN neighbors")
    parser.add_argument('--test_ratio', type=float, default=0.2, help="Test set ratio")
    args = parser.parse_args()
    
    print("Loading and combining datasets...")
    df_combined = load_and_combine(args.files)
    df_combined = add_time_features(df_combined)
    
    print(f"Total combined rows: {len(df_combined)}")
    train_knn(df_combined, n_neighbors=args.neighbors, test_ratio=args.test_ratio)
