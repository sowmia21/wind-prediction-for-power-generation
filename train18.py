import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

# -----------------------------
# Load and preprocess dataset
# -----------------------------
data_path = "wind data 1.csv"
print(f"Loading dataset: {data_path}")

df = pd.read_csv(data_path).dropna()

# Handle datetime column if exists
for col in df.columns:
    if df[col].dtype == 'object':
        try:
            df[col] = pd.to_datetime(df[col])
            df[col + '_hour'] = df[col].dt.hour
            df[col + '_day'] = df[col].dt.day
            df[col + '_month'] = df[col].dt.month
            df[col + '_weekday'] = df[col].dt.weekday
            df.drop(columns=[col], inplace=True)
        except:
            df.drop(columns=[col], inplace=True)

X = df.drop(columns=[df.columns[-1]]).values
y = df[df.columns[-1]].values

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print(f"Dataset shape: {df.shape}")
print(f"Features used for training: {df.drop(columns=[df.columns[-1]]).columns.tolist()}")

# -----------------------------
# Hyperparameter search space
# -----------------------------
space = {
    'n_estimators': hp.choice('n_estimators', range(50, 501, 50)),
    'max_depth': hp.choice('max_depth', range(3, 15)),
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
    'subsample': hp.uniform('subsample', 0.6, 1.0),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0)
}

# -----------------------------
# Objective function
# -----------------------------
def objective(params):
    model = XGBRegressor(
        n_estimators=int(params['n_estimators']),
        max_depth=int(params['max_depth']),
        learning_rate=params['learning_rate'],
        subsample=params['subsample'],
        colsample_bytree=params['colsample_bytree'],
        random_state=42,
        n_jobs=-1
    )
    # 3-fold CV RMSE
    score = cross_val_score(model, X_train, y_train, cv=3,
                            scoring='neg_root_mean_squared_error').mean()
    return {'loss': -score, 'status': STATUS_OK}

# -----------------------------
# Run Hyperopt
# -----------------------------
trials = Trials()
best = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest,
    max_evals=50,  # number of iterations
    trials=trials,
    rstate=np.random.default_rng(42)
)

print("\nBest hyperparameters:", best)

# -----------------------------
# Train final model with best params
# -----------------------------
best_model = XGBRegressor(
    n_estimators=int(best['n_estimators']),
    max_depth=int(best['max_depth']),
    learning_rate=best['learning_rate'],
    subsample=best['subsample'],
    colsample_bytree=best['colsample_bytree'],
    random_state=42,
    n_jobs=-1
)
best_model.fit(X_train, y_train)

# Predict and evaluate
y_pred = best_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Test RMSE: {rmse:.4f}")
