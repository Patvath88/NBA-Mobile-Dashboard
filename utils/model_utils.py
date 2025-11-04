# utils/model_utils.py
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path

# ------------------------------
# 🔹 CONFIG
# ------------------------------
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = ["PTS", "REB", "AST", "STL", "BLK", "TOV", "FG3M", "PRA"]


# ------------------------------
# 🔹 MODEL TRAINING
# ------------------------------

def train_xgboost_models(feature_df: pd.DataFrame):
    """
    Train an individual XGBoost regressor for each stat category.
    Save models in /models/.
    """
    results = {}

    # Independent features: all numeric except target stats
    X = feature_df.drop(columns=TARGETS, errors="ignore").select_dtypes(include=np.number)

    for target in TARGETS:
        if target not in feature_df.columns:
            continue

        y = feature_df[target]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        )

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = mean_squared_error(y_test, preds, squared=False)
        r2 = r2_score(y_test, preds)

        results[target] = {"MAE": mae, "RMSE": rmse, "R2": r2}

        # Save model
        model_path = MODEL_DIR / f"xgb_{target.lower()}.pkl"
        joblib.dump(model, model_path)

    return results


# ------------------------------
# 🔹 PREDICTION
# ------------------------------

def predict_next_game(feature_df: pd.DataFrame):
    """
    Predict next game stats using saved models.
    Takes the latest game row as input.
    """
    latest_row = feature_df.tail(1).select_dtypes(include=np.number)
    preds = {}

    for target in TARGETS:
        model_path = MODEL_DIR / f"xgb_{target.lower()}.pkl"
        if not model_path.exists():
            preds[target] = np.nan
            continue

        model = joblib.load(model_path)
        pred_value = float(model.predict(latest_row)[0])
        preds[target] = round(pred_value, 1)

    preds["PRA"] = round(preds.get("PTS", 0) + preds.get("REB", 0) + preds.get("AST", 0), 1)
    return preds


# ------------------------------
# 🔹 EVALUATION (For Dashboard)
# ------------------------------

def evaluate_model_performance(feature_df: pd.DataFrame):
    """Compute cross-validated performance summary for dashboard display."""
    X = feature_df.drop(columns=TARGETS, errors="ignore").select_dtypes(include=np.number)
    eval_summary = {}

    for target in TARGETS:
        if target not in feature_df.columns:
            continue
        y = feature_df[target]
        model_path = MODEL_DIR / f"xgb_{target.lower()}.pkl"
        if not model_path.exists():
            continue

        model = joblib.load(model_path)
        preds = model.predict(X)
        eval_summary[target] = {
            "MAE": round(mean_absolute_error(y, preds), 2),
            "RMSE": round(mean_squared_error(y, preds, squared=False), 2),
            "R2": round(r2_score(y, preds), 2)
        }

    return pd.DataFrame(eval_summary).T
