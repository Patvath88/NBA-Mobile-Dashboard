import streamlit as st
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ------------------------------------------------------
# ⚙️ Training and Caching
# ------------------------------------------------------
@st.cache_resource(ttl=604800)
def load_or_train_cached_model(df: pd.DataFrame):
    """Train lightweight models weekly and cache them."""
    if df is None or df.empty:
        return {}

    models = {}
    X = df.select_dtypes(include=[np.number])
    targets = [t for t in ["PTS", "REB", "AST"] if t in X.columns]
    X = X.drop(columns=targets, errors="ignore")

    for target in targets:
        y = df[target]
        if len(X) < 10:
            continue
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        model = xgb.XGBRegressor(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.9,
            tree_method="hist",
            random_state=42,
            verbosity=0
        )
        model.fit(X_train, y_train)
        models[target] = model
    return models


# ------------------------------------------------------
# 🧠 Model Training
# ------------------------------------------------------
@st.cache_data(ttl=1800)
def train_xgboost_models(df: pd.DataFrame):
    """Evaluate cached mini-model performance."""
    models = load_or_train_cached_model(df)
    if not models:
        st.warning("No model trained (insufficient data).")
        return {}

    results = {}
    for target, model in models.items():
        X = df.select_dtypes(include=[np.number]).drop(columns=[target], errors="ignore")
        y = df[target]
        preds = model.predict(X)
        rmse = mean_squared_error(y, preds, squared=False)
        mae = mean_absolute_error(y, preds)
        results[target] = {"RMSE": round(rmse, 2), "MAE": round(mae, 2), "MeanPred": float(np.mean(preds))}
    return results


# ------------------------------------------------------
# 🎯 Prediction
# ------------------------------------------------------
def predict_next_game(df, models=None):
    """
    Predict next game stats with feature alignment to prevent XGBoost mismatch errors.
    """
    import numpy as np
    import streamlit as st

    try:
        if df is None or df.empty:
            st.error("No valid data for prediction.")
            return {}

        # Drop non-numeric columns if any
        df_numeric = df.select_dtypes(include=[np.number]).copy()
        if df_numeric.empty:
            st.error("No numeric features available for prediction.")
            return {}

        # Use only last game for context
        latest_features = df_numeric.tail(1).copy()

        # Align feature columns with trained model
        if models and "PTS" in models:
            expected_features = models["PTS"].get_booster().feature_names
            if expected_features:
                available_features = latest_features.columns.tolist()
                missing_cols = [col for col in expected_features if col not in available_features]
                extra_cols = [col for col in available_features if col not in expected_features]

                # Add missing columns with default 0
                for col in missing_cols:
                    latest_features[col] = 0

                # Drop extra columns not used during training
                latest_features = latest_features[[c for c in expected_features if c in latest_features.columns]]

        preds = {}
        for stat, model in models.items():
            pred_value = model.predict(latest_features)[0]
            preds[stat] = round(float(pred_value), 1)

        return preds

    except Exception as e:
        st.error(f"Error during prediction alignment: {e}")
        return {}

