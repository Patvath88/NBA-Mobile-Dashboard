# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
from utils.data_loader import get_player_context, get_player_id
from utils.feature_engineer import build_feature_dataset
from utils.model_utils import train_xgboost_models, predict_next_game
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_lottie import st_lottie
import requests
import warnings
warnings.filterwarnings("ignore")

# ======================================================
# 🧠 APP CONFIG
# ======================================================
st.set_page_config(
    page_title="🏀 NBA Predictive Dashboard — FutureCourt Edition",
    layout="wide",
    page_icon="🏀",
    initial_sidebar_state="collapsed"
)

# ======================================================
# 🎨 LOAD CUSTOM CSS
# ======================================================
def load_custom_css():
    css = """
    /* =============== FUTURECOURT THEME =============== */
    body {
        background: radial-gradient(circle at 25% 25%, #0d0d0f, #050505 80%);
        color: #f8f8f8;
        font-family: 'Inter', sans-serif;
    }
    h1, h2, h3, h4 {
        font-weight: 700;
        text-transform: uppercase;
        color: #f7a600;
        letter-spacing: 0.5px;
    }
    .stApp {
        background-color: transparent !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 2.2rem !important;
        font-weight: 700;
        color: #ffb347 !important;
        text-shadow: 0 0 20px #ffb34788;
    }
    [data-testid="stMetricLabel"] {
        text-transform: uppercase;
        color: #aaa !important;
        font-size: 0.8rem;
    }
    div.stButton > button {
        background: linear-gradient(135deg, #f97316, #facc15);
        color: black !important;
        border-radius: 8px;
        font-weight: 700;
        transition: all 0.3s ease;
        border: none;
    }
    div.stButton > button:hover {
        background: linear-gradient(135deg, #facc15, #f97316);
        transform: scale(1.05);
        box-shadow: 0 0 20px #facc1577;
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    /* ✨ Fade In Animation */
    @keyframes fadeIn {
        from {opacity: 0; transform: translateY(10px);}
        to {opacity: 1; transform: translateY(0);}
    }
    .main {
        animation: fadeIn 0.9s ease-in-out;
    }
    /* ✨ Pulsing Glow Background */
    @keyframes pulse {
        0% { background: radial-gradient(circle at 50% 50%, #facc1555, transparent 70%); }
        50% { background: radial-gradient(circle at 50% 50%, #f9731655, transparent 70%); }
        100% { background: radial-gradient(circle at 50% 50%, #facc1555, transparent 70%); }
    }
    body::before {
        content: "";
        position: fixed;
        top: 0; left: 0;
        width: 100%; height: 100%;
        z-index: -1;
        animation: pulse 12s infinite;
        background-size: 200% 200%;
    }
    """
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

load_custom_css()

# ======================================================
# 🏀 HEADER
# ======================================================
st.title("🏀 NBA Predictive Dashboard — FutureCourt Edition")
st.caption("Real-time Player Insights • Predictive Modeling • AI-Powered Analysis")

# Lottie animation (basketball)
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_url = "https://assets5.lottiefiles.com/packages/lf20_jyjjgplv.json"
st_lottie(load_lottieurl(lottie_url), height=120, key="lottie-basketball")

# ======================================================
# 🔍 PLAYER SEARCH
# ======================================================
player_name = st.text_input("Enter Player Name (e.g. Luka Doncic)", "")
opponent = st.text_input("Next Opponent Team Name (e.g. Boston Celtics)", "")

# ======================================================
# ⚙️ FETCH PLAYER DATA
# ======================================================
if st.button("Fetch Player Data"):
    with st.spinner("⏳ Fetching player stats and context..."):
        try:
            context = get_player_context(player_name, opponent)
            if not context:
                st.error("Unable to load player data.")
            else:
                st.subheader("📊 Season Averages")
                st.dataframe(pd.DataFrame(context["season_avg"]).T)

                # Plot last 10 games trend
                st.subheader("📈 Recent Trends (Last 10 Games)")
                df = context["recent_games"]

                if not df.empty and "GAME_DATE" in df.columns:
                    fig = px.line(
                        df,
                        x="GAME_DATE",
                        y=["PTS", "REB", "AST", "PRA"],
                        markers=True,
                        title=f"{player_name} Performance Trends",
                    )
                    fig.update_layout(
                        template="plotly_dark",
                        plot_bgcolor="rgba(10,10,10,0.6)",
                        paper_bgcolor="rgba(10,10,10,0.6)",
                        font=dict(color="#f8f8f8"),
                        xaxis_title="Game Date",
                        yaxis_title="Stat Value",
                        hovermode="x unified",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No recent game data available.")
        except Exception as e:
            st.error(f"Error fetching player data: {e}")

# ======================================================
# 🤖 MODEL TRAINING + PREDICTION
# ======================================================
if st.button("Train & Predict Next Game"):
    with st.spinner("🤖 Training AI model and generating predictions..."):
        try:
            player_id = get_player_id(player_name)
            df = build_feature_dataset(player_id)
            results = train_xgboost_models(df)
            preds = predict_next_game(df)

            st.subheader("🔮 Next Game Predictions")

            col1, col2, col3 = st.columns(3)
            col1.metric("Points", f"{preds.get('PTS', 0):.1f}")
            col2.metric("Rebounds", f"{preds.get('REB', 0):.1f}")
            col3.metric("Assists", f"{preds.get('AST', 0):.1f}")
            style_metric_cards(
                background_color="#1c1c1e",
                border_left_color="#f97316",
                border_color="#facc15",
                box_shadow=True,
            )

            st.subheader("📉 Model Evaluation Summary")
            st.dataframe(pd.DataFrame(results).T, use_container_width=True)

        except Exception as e:
            st.error(f"Error during model training or prediction: {e}")

# ======================================================
# 🛰️ FOOTER
# ======================================================
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#777;'>⚙️ Powered by XGBoost • NBA API • Streamlit • FutureCourt Design</p>",
    unsafe_allow_html=True,
)
