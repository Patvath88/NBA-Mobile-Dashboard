import xgboost as xgb
from sklearn.model_selection import train_test_split
import numpy as np
import streamlit as st

@st.cache_resource(ttl=7200)
def train_xgboost_models(dataset):
    target_cols = ["PTS", "REB", "AST", "PRA"]
    features = [c for c in dataset.columns if c not in target_cols + ["GAME_DATE", "Opponent", "TEAM_NAME"]]
    models = {}

    for target in target_cols:
        X_train, X_test, y_train, y_test = train_test_split(dataset[features], dataset[target], test_size=0.2, random_state=42)
        model = xgb.XGBRegressor(n_estimators=80, learning_rate=0.08, max_depth=4, subsample=0.9, colsample_bytree=0.9)
        model.fit(X_train, y_train)
        models[target] = model
    return models

def predict_next_game(models, dataset):
    last_row = dataset.iloc[[-1]]
    preds = {stat: float(model.predict(last_row.drop(columns=["PTS", "REB", "AST", "PRA"], errors="ignore"))[0]) for stat, model in models.items()}
    return preds
