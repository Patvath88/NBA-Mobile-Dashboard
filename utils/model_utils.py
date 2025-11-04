import xgboost as xgb
from sklearn.model_selection import train_test_split
import numpy as np
import streamlit as st

@st.cache_resource(ttl=7200)
def train_xgboost_models(df):
    """
    Trains XGBoost regressors for NBA player prediction.
    Handles small datasets gracefully.
    """
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    import numpy as np
    import pandas as pd
    import streamlit as st

    if df is None or df.empty:
        st.warning("No data available for training.")
        return None

    # Ensure minimum sample count
    if len(df) < 5:
        st.warning("Not enough game samples to train a reliable model.")
        return None

    targets = ["PTS", "REB", "AST", "PRA"]
    models = {}

    feature_cols = [col for col in df.columns if col not in targets and df[col].dtype in [int, float]]
    X = df[feature_cols]
    y_data = df[targets]

    for target in targets:
        y = df[target]

        try:
            # ✅ Safe train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2 if len(df) >= 10 else 0.1, random_state=42
            )

            if len(X_train) == 0 or len(X_test) == 0:
                st.warning(f"Not enough data to train model for {target}.")
                continue

            model = xgb.XGBRegressor(
                n_estimators=200,
                learning_rate=0.08,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )

            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, preds))

            models[target] = model
            print(f"✅ Trained {target} model (RMSE={rmse:.2f})")

        except Exception as e:
            print(f"⚠️ Error training model for {target}: {e}")

    return models if models else None


def predict_next_game(models, df):
    """
    Generates predictions for the next game using trained XGBoost models.
    Returns a dictionary of predictions or empty results if models invalid.
    """
    import pandas as pd

    if not models or df is None or df.empty:
        print("⚠️ Prediction skipped — no trained models or valid data.")
        return {"PTS": 0, "REB": 0, "AST": 0, "PRA": 0}

    last_row = df.tail(1)

    preds = {}
    for stat in ["PTS", "REB", "AST", "PRA"]:
        model = models.get(stat)
        if model is None:
            preds[stat] = 0
            continue

        X_input = last_row.drop(columns=["PTS", "REB", "AST", "PRA"], errors="ignore")
        preds[stat] = float(model.predict(X_input)[0])

    return preds
