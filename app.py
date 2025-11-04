import streamlit as st
import pandas as pd
import plotly.express as px
from utils.data_loader import get_player_context, get_player_id
from utils.feature_engineer import build_feature_dataset
from utils.model_utils import train_xgboost_models, predict_next_game
from utils.helpers import sanitize_dataframe_for_streamlit
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_lottie import st_lottie
import requests
import warnings

warnings.filterwarnings("ignore")

# ======================================================
# 🌌 FUTURECOURT DASHBOARD THEME & PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="FutureCourt NBA AI",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Inject sleek dark theme CSS
st.markdown("""
    <style>
        body {
            background-color: #0b0c10;
            color: #f1f1f1;
        }
        .stApp {
            background: radial-gradient(circle at 20% 20%, #111 0%, #000 100%);
            color: #f1f1f1;
        }
        .block-container {
            padding: 2rem 1rem;
        }
        .stTextInput>div>div>input {
            background-color: #111827;
            color: #f1f1f1;
            border: 1px solid #f97316;
            border-radius: 8px;
        }
        .stButton button {
            background: linear-gradient(90deg, #f59e0b, #f97316);
            border: none;
            border-radius: 8px;
            color: white;
            font-weight: 600;
            transition: 0.3s ease-in-out;
        }
        .stButton button:hover {
            transform: scale(1.05);
            background: linear-gradient(90deg, #fbbf24, #f97316);
        }
        h1, h2, h3, h4 {
            color: #fbbf24 !important;
        }
    </style>
""", unsafe_allow_html=True)

# ======================================================
# 🏀 HEADER + ANIMATION
# ======================================================
def load_lottie(url: str):
    try:
        res = requests.get(url)
        if res.status_code == 200:
            return res.json()
    except Exception:
        return None
    return None

lottie_nba = load_lottie("https://assets3.lottiefiles.com/packages/lf20_2glqweqs.json")

col1, col2 = st.columns([3, 1])
with col1:
    st.title("🏀 FutureCourt NBA Prediction Dashboard")
    st.caption("AI-powered player analytics & predictive modeling in real-time")
with col2:
    st_lottie(lottie_nba, height=100, key="nba_lottie")

st.divider()

# ======================================================
# 🔍 PLAYER SEARCH INPUT
# ======================================================
player_name = st.text_input("Enter Player Name (e.g. Luka Doncic)", "")

# ======================================================
# ⚙️ FETCH PLAYER DATA
# ======================================================
if st.button("Fetch Player Data"):
    with st.spinner("⏳ Fetching player stats and context..."):
        try:
            context = get_player_context(player_name)
            if context:
                st.subheader("📊 Season Averages")
                df = sanitize_dataframe_for_streamlit(pd.DataFrame(context["season_avg"]).T)
                st.dataframe(df, use_container_width=True)

                st.subheader("📈 Last 10 Games Trend")
                recent = sanitize_dataframe_for_streamlit(context["recent_games"])
                if not recent.empty and "GAME_DATE" in recent.columns:
                    fig = px.line(
                        recent,
                        x="GAME_DATE",
                        y=["PTS", "REB", "AST", "PRA"],
                        title=f"{player_name} - Performance Trends (Last 10 Games)",
                        markers=True,
                    )
                    fig.update_layout(
                        template="plotly_dark",
                        height=400,
                        margin=dict(l=0, r=0, t=40, b=0),
                        plot_bgcolor="#0b0c10",
                        paper_bgcolor="#0b0c10",
                        font=dict(color="#f1f1f1")
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No recent game data found.")
        except Exception as e:
            st.error(f"Error building player context: {e}")

st.divider()

# ======================================================
# 🤖 MODEL TRAINING & PREDICTION
# ======================================================
if st.button("Train & Predict Next Game"):
    with st.spinner("🧠 Training and predicting..."):
        try:
            player_id = get_player_id(player_name)
            df = build_feature_dataset(player_id)

            if df is not None and not df.empty:
                results = train_xgboost_models(df)
                preds = predict_next_game(df)

                col1, col2, col3 = st.columns(3)
                col1.metric("Predicted Points", round(preds.get("PTS", 0), 1))
                col2.metric("Predicted Rebounds", round(preds.get("REB", 0), 1))
                col3.metric("Predicted Assists", round(preds.get("AST", 0), 1))
                style_metric_cards(background_color="#111827", border_color="#f59e0b", border_left_color="#f97316")

                st.divider()
                st.subheader("📊 Model Diagnostics")
                res_df = sanitize_dataframe_for_streamlit(pd.DataFrame(results))
                st.dataframe(res_df, use_container_width=True)
            else:
                st.warning("Insufficient data to train model for this player.")
        except Exception as e:
            st.error(f"Error during model training or prediction: {e}")

st.caption("⚡ Powered by XGBoost · Streamlit · NBA API · FutureCourt UI")
