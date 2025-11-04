import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import streamlit as st


def train_xgboost_models(df):
    """Train XGBoost models for PTS, REB, AST."""
    try:
        target_stats = ["PTS", "REB", "AST"]
        feature_cols = [c for c in df.columns if c not in ["GAME_DATE", "PTS", "REB", "AST", "MATCHUP", "WL"]]

        models = {}
        results = {}

        for stat in target_stats:
            X = df[feature_cols]
            y = df[stat]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

            model = xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
            )
            model.fit(X_train, y_train)

            preds = model.predict(X_test)
            results[stat] = {
                "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
                "MAE": mean_absolute_error(y_test, preds),
                "R²": r2_score(y_test, preds),
            }
            models[stat] = model

        return results | {"models": models}

    except Exception as e:
        st.error(f"Error during model training: {e}")
        return {}


def predict_next_game(df, models=None):
    """Predict next game stats with feature alignment."""
    try:
        if not models or "models" not in models:
            st.error("No trained models available for prediction.")
            return {}

        models = models["models"]

        # Use last available record
        latest_features = df.select_dtypes(include=[np.number]).tail(1).copy()

        preds = {}
        for stat, model in models.items():
            expected_features = model.get_booster().feature_names
            missing_cols = [c for c in expected_features if c not in latest_features.columns]
            for c in missing_cols:
                latest_features[c] = 0
            latest_features = latest_features[expected_features]
            pred_val = float(model.predict(latest_features)[0])
            preds[stat] = round(pred_val, 1)

        return preds
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return {}
