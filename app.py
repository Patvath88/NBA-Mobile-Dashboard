# app.py
import streamlit as st
import pandas as pd
from utils.data_loader import get_player_context, get_player_id
from utils.feature_engineer import build_feature_dataset
from utils.model_utils import train_xgboost_models, predict_next_game, evaluate_model_performance
import plotly.express as px

# ------------------------------
# ⚙️ PAGE CONFIG
# ------------------------------
st.set_page_config(
    page_title="🏀 Hot Shot NBA Predictor",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------
# 🎨 CUSTOM CSS (Dark Mode Theme)
# ------------------------------
st.markdown("""
    <style>
    body { background-color: #0D0D0D; color: #F5F5F5; }
    [data-testid="stSidebar"] { background-color: #111; }
    h1, h2, h3, h4 { color: #FFD700; }
    .stButton>button {
        background-color: #F97316;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #FFB84D;
        color: black;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------------------
# 🔹 SIDEBAR NAVIGATION
# ------------------------------
page = st.sidebar.radio(
    "📊 Navigate",
    ["🏀 Player Viewer", "🧠 Model Predictor", "📈 Live Tracker"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Powered by 🧠 Code GPT + XGBoost")

# ------------------------------
# 🧍 PLAYER VIEWER
# ------------------------------
if page == "🏀 Player Viewer":
    st.title("🏀 Player Performance Viewer")

    player_name = st.text_input("Enter Player Name:", "LeBron James")
    opponent = st.text_input("Next Opponent Team:", "Boston Celtics")

    if st.button("Fetch Player Data"):
        with st.spinner("Loading data..."):
            context = get_player_context(player_name, opponent)

            st.subheader("📊 Season Averages")
            st.dataframe(pd.DataFrame(context["season_avg"]).T)

            st.subheader("📈 Last 10 Games Trend")
            df = context["recent_games"]

            fig = px.line(
                df,
                x="GAME_DATE",
                y=["PTS", "REB", "AST", "PRA"],
                title=f"{player_name} - Recent Game Trends",
                template="plotly_dark",
                markers=True
            )
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("🛡️ Opponent Defensive Metrics")
            st.json(context["opponent_defense"])

# ------------------------------
# 🧠 MODEL PREDICTOR
# ------------------------------
elif page == "🧠 Model Predictor":
    st.title("🧠 Predictive Modeling Engine (XGBoost)")

    player = st.text_input("Enter Player Name:", "Luka Doncic")

    if st.button("Train & Predict Next Game"):
        with st.spinner("Training and predicting..."):
            player_id = get_player_id(player)
            df = build_feature_dataset(player_id)

            results = train_xgboost_models(df)
            preds = predict_next_game(df)
            eval_df = evaluate_model_performance(df)

            st.subheader("📊 Model Training Summary")
            st.dataframe(pd.DataFrame(results).T)

            st.subheader("🎯 Predicted Next Game Stats")
            st.json(preds)

            st.subheader("📈 Model Evaluation")
            st.dataframe(eval_df)

            # Visualization of predicted performance
            fig = px.bar(
                x=list(preds.keys()),
                y=list(preds.values()),
                color=list(preds.keys()),
                title=f"{player} - Predicted Next Game Output",
                template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)

# ------------------------------
# 📈 LIVE TRACKER (AUTO REFRESH)
# ------------------------------
elif page == "📈 Live Tracker":
    st.title("📈 Live Prediction Accuracy Tracker (30s Auto Refresh)")

    st.markdown(
        "Auto-refreshing every **30 seconds** to compare predicted vs actual performance."
    )

    player_name = st.text_input("Enter Player Name:", "Jayson Tatum")

    if player_name:
        import requests
        from utils.model_utils import predict_next_game
        from utils.feature_engineer import build_feature_dataset

        player_id = get_player_id(player_name)
        feature_df = build_feature_dataset(player_id)
        predicted = predict_next_game(feature_df)

        st_autorefresh = st.autorefresh(interval=30 * 1000, key="refresh")

        live_url = f"https://www.balldontlie.io/api/v1/stats?player_ids[]={player_id}"
        res = requests.get(live_url).json()

        if len(res["data"]) == 0:
            st.warning("No live game data available.")
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

            df_compare = pd.DataFrame([
                {"Stat": k, "Predicted": predicted.get(k, 0), "Actual": actual.get(k, 0)}
                for k in predicted.keys()
            ])
            df_compare["Error"] = abs(df_compare["Predicted"] - df_compare["Actual"])
            df_compare["% Error"] = (df_compare["Error"] / (df_compare["Actual"] + 1e-6) * 100).round(1)

            st.dataframe(df_compare, use_container_width=True)

            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_trace(go.Bar(x=df_compare["Stat"], y=df_compare["Predicted"], name="Predicted"))
            fig.add_trace(go.Bar(x=df_compare["Stat"], y=df_compare["Actual"], name="Actual"))
            fig.update_layout(barmode="group", template="plotly_dark", title="Predicted vs Actual (Live)")
            st.plotly_chart(fig, use_container_width=True)

            st.metric("MAE", df_compare["Error"].mean().round(2))
            st.metric("Average % Error", f"{df_compare['% Error'].mean().round(1)}%")
