# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
from utils.data_loader import get_player_context, get_player_id
from utils.feature_engineer import build_feature_dataset
from utils.model_utils import train_xgboost_models, predict_next_game, evaluate_model_performance
import warnings

warnings.filterwarnings("ignore")

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
# 🎨 CUSTOM DARK THEME CSS
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
# 🧹 HELPER FUNCTIONS
# ------------------------------
def sanitize_df(df: pd.DataFrame):
    """Ensure DataFrame is Arrow-safe for Streamlit."""
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = df[c].astype(str)
        if df[c].dtype == "object":
            df[c] = df[c].apply(lambda x: str(x) if not isinstance(x, (int, float)) else x)
    return df


def safe_plot(df, player_name):
    """Safe plot function for player stat trends."""
    if "GAME_DATE" not in df.columns:
        st.warning("No GAME_DATE column found — unable to plot trends.")
        return
    if not all(c in df.columns for c in ["PTS", "REB", "AST"]):
        st.warning("Missing stat columns for plotting.")
        return

    # Create PRA column
    df["PRA"] = df["PTS"] + df["REB"] + df["AST"]
    for col in ["PTS", "REB", "AST", "PRA"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    fig = px.line(
        df,
        x="GAME_DATE",
        y=["PTS", "REB", "AST", "PRA"],
        title=f"{player_name} - Recent Game Trends",
        template="plotly_dark",
        markers=True
    )
    st.plotly_chart(fig, width="stretch")


# ------------------------------
# 🧭 SIDEBAR NAVIGATION
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
            season_avg_df = pd.DataFrame(context.get("season_avg", {})).T
            st.dataframe(sanitize_df(season_avg_df), width="stretch")

            st.subheader("📈 Last 10 Games Trend")
            df = context.get("recent_games", pd.DataFrame())
            if df.empty:
                st.warning("No recent games found for this player.")
            else:
                safe_plot(df.copy(), player_name)

            st.subheader("🛡️ Opponent Defensive Metrics")
            opp_def = context.get("opponent_metrics", context.get("opponent_defense", {}))
            if isinstance(opp_def, dict) and opp_def:
                st.json(opp_def)
            else:
                st.info("No opponent defensive data available.")

# ------------------------------
# 🧠 MODEL PREDICTOR
# ------------------------------
elif page == "🧠 Model Predictor":
    st.title("🧠 Predictive Modeling Engine (XGBoost)")
    player = st.text_input("Enter Player Name:", "Luka Doncic")

    if st.button("Train & Predict Next Game"):
        with st.spinner("Training and predicting..."):
            try:
                player_id = get_player_id(player)
                df = build_feature_dataset(player_id)

                if df.empty:
                    st.warning("No player data available for training.")
                    st.stop()

                results = train_xgboost_models(df)
                preds = predict_next_game(df)
                eval_df = evaluate_model_performance(df)

                st.subheader("📊 Model Training Summary")
                st.dataframe(sanitize_df(pd.DataFrame(results).T), width="stretch")

                st.subheader("🎯 Predicted Next Game Stats")
                st.json(preds)

                st.subheader("📈 Model Evaluation")
                st.dataframe(sanitize_df(eval_df), width="stretch")

                fig = px.bar(
                    x=list(preds.keys()),
                    y=list(preds.values()),
                    title=f"{player} - Predicted Next Game Output",
                    template="plotly_dark"
                )
                st.plotly_chart(fig, width="stretch")

            except Exception as e:
                st.error(f"Error during model training or prediction: {e}")

# ------------------------------
# 📈 LIVE TRACKER (SIMPLE)
# ------------------------------
elif page == "📈 Live Tracker":
    st.title("📈 Live Prediction Accuracy Tracker (Simplified)")
    st.info("For full live tracking, integrate with /pages/3_Live_Tracker.py")
