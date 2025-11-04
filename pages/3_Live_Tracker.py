# pages/3_Live_Tracker.py
import streamlit as st
import pandas as pd
import requests
import time
from utils.data_loader import get_player_id
from utils.feature_engineer import build_feature_dataset
from utils.model_utils import predict_next_game
import plotly.graph_objects as go

# ------------------------------
# ⚙️ PAGE CONFIG
# ------------------------------
st.set_page_config(
    page_title="📈 Live Prediction Tracker",
    layout="wide",
    page_icon="🏀",
    initial_sidebar_state="expanded"
)

st.title("📈 Live NBA Prediction Accuracy Tracker")
st.markdown("Auto-refreshes every **30 seconds** for real-time updates ⏱️")

# ------------------------------
# 🔁 AUTO REFRESH
# ------------------------------
st_autorefresh = st.experimental_rerun  # legacy fallback
count = st.experimental_rerun if callable(st.experimental_rerun) else None
st_autorefresh = st_autorefresh or st.autorefresh
st_autorefresh(interval=30 * 1000, key="data_refresh")  # 30s

# ------------------------------
# 🔹 USER INPUT
# ------------------------------
player_name = st.text_input("Enter Player Name:", "LeBron James")
season = "2024-25"

if player_name:
    try:
        player_id = get_player_id(player_name)

        with st.spinner("Fetching predictions and live data..."):
            # Predict next game stats
            feature_df = build_feature_dataset(player_id, season)
            predicted = predict_next_game(feature_df)

            # Fetch live game data
            live_url = f"https://www.balldontlie.io/api/v1/stats?player_ids[]={player_id}"
            res = requests.get(live_url).json()

            if len(res["data"]) == 0:
                st.warning("No active or recent live data for this player.")
            else:
                live_game = res["data"][-1]["stats"]
                actual = {
                    "PTS": live_game.get("pts", 0),
                    "REB": live_game.get("reb", 0),
                    "AST": live_game.get("ast", 0),
                    "STL": live_game.get("stl", 0),
                    "BLK": live_game.get("blk", 0),
                    "TOV": live_game.get("turnover", 0),
                    "FG3M": live_game.get("fg3m", 0)
                }
                actual["PRA"] = actual["PTS"] + actual["REB"] + actual["AST"]

                # Compute accuracy
                df_compare = pd.DataFrame([
                    {"Stat": k, "Predicted": predicted.get(k, 0), "Actual": actual.get(k, 0)}
                    for k in predicted.keys()
                ])
                df_compare["Error"] = abs(df_compare["Predicted"] - df_compare["Actual"])
                df_compare["% Error"] = (df_compare["Error"] / (df_compare["Actual"] + 1e-6) * 100).round(1)

                st.subheader(f"🎯 {player_name} – Live Performance Tracker")
                st.dataframe(df_compare, use_container_width=True)

                # Live comparison chart
                fig = go.Figure()
                fig.add_trace(go.Bar(x=df_compare["Stat"], y=df_compare["Predicted"], name="Predicted"))
                fig.add_trace(go.Bar(x=df_compare["Stat"], y=df_compare["Actual"], name="Actual"))
                fig.update_layout(
                    title=f"{player_name} – Predicted vs Actual",
                    barmode="group",
                    template="plotly_dark",
                    height=400,
                    margin=dict(l=20, r=20, t=50, b=20)
                )
                st.plotly_chart(fig, use_container_width=True)

                # Metrics summary
                mae = df_compare["Error"].mean().round(2)
                avg_err = df_compare["% Error"].mean().round(1)
                st.metric(label="Mean Absolute Error (MAE)", value=mae)
                st.metric(label="Average % Error", value=f"{avg_err}%")

    except Exception as e:
        st.error(f"Error: {e}")
