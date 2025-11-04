# utils/model_utils.py
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import streamlit as st

# ------------------------------
# 🧠 TRAIN MODEL
# ------------------------------
def train_xgboost_models(df: pd.DataFrame):
    """Train an XGBoost model for each target stat."""
    targets = ["PTS", "REB", "AST", "STL", "BLK", "TOV", "FG3M", "PRA"]
    results = {}

    try:
        for target in targets:
            if target not in df.columns:
                continue

            X = df.drop(columns=[target], errors="ignore")
            y = df[target]

            if len(X) < 5:
                continue  # Not enough games

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, shuffle=False
            )

            model = xgb.XGBRegressor(
                n_estimators=250,
                learning_rate=0.08,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
            )
            model.fit(X_train, y_train)

            preds = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            mae = mean_absolute_error(y_test, preds)
            r2 = r2_score(y_test, preds)

            results[target] = {
                "RMSE": round(rmse, 2),
                "MAE": round(mae, 2),
                "R2": round(r2, 3),
                "Games Trained": len(X_train),
            }

        return results

    except Exception as e:
        st.error(f"Error training XGBoost models: {e}")
        return {}


# ------------------------------
# 🎯 PREDICT NEXT GAME
# ------------------------------
def predict_next_game(df: pd.DataFrame):
    """Predict next game stats using the last available data row."""
    preds = {}
    try:
        last_row = df.iloc[-1:].drop(columns=["PRA"], errors="ignore")
        for target in ["PTS", "REB", "AST", "STL", "BLK", "TOV", "FG3M"]:
            model = xgb.XGBRegressor()
            model.fit(df.drop(columns=[target], errors="ignore"), df[target])
            preds[target] = float(model.predict(last_row)[0])
        preds["PRA"] = preds["PTS"] + preds["REB"] + preds["AST"]
        return preds
    except Exception as e:
        st.error(f"Error predicting next game: {e}")
        return {}


# ------------------------------
# 📈 EVALUATE MODEL PERFORMANCE
# ------------------------------
def evaluate_model_performance(df: pd.DataFrame):
    """Compute RMSE, MAE, and R² for all major stats."""
    metrics = []
    try:
        for stat in ["PTS", "REB", "AST", "PRA"]:
            if stat not in df.columns:
                continue
            y = df[stat]
            y_pred = df[stat].rolling(1).mean()  # simple baseline
            mse = mean_squared_error(y[1:], y_pred[1:])
            rmse = float(np.sqrt(mse))
            mae = float(mean_absolute_error(y[1:], y_pred[1:]))
            r2 = float(r2_score(y[1:], y_pred[1:]))
            metrics.append({
                "Stat": stat,
                "RMSE": round(rmse, 2),
                "MAE": round(mae, 2),
                "R2": round(r2, 3)
            })

        return pd.DataFrame(metrics)

    except Exception as e:
        st.error(f"Error evaluating model performance: {e}")
        return pd.DataFrame()
