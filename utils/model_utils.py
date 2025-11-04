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
def predict_next_game(df: pd.DataFrame):
    models = load_or_train_cached_model(df)
    if not models:
        return {}
    last = df.tail(1).select_dtypes(include=[np.number])
    preds = {t: float(models[t].predict(last)[0]) for t in models}
    return preds
