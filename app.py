# ======================================================
# 🏀 FUTURECOURT NBA AI — Futuristic Dashboard
# ======================================================

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

st.markdown(
    """
    <style>
    body {
        background: linear-gradient(120deg, #0f2027, #203a43, #2c5364);
        color: white;
        font-family: 'Segoe UI', sans-serif;
    }
    .stApp {
        background: transparent;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 1rem;
        box-shadow: 0 0 10px rgba(0,0,0,0.3);
        transition: all 0.3s ease-in-out;
    }
    .metric-card:hover {
        transform: scale(1.02);
        box-shadow: 0 0 20px rgba(255,255,255,0.2);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ======================================================
# 🧠 HEADER & ANIMATION
# ======================================================

col1, col2 = st.columns([0.7, 0.3])
with col1:
    st.title("🏀 FutureCourt NBA AI Dashboard")
    st.markdown("#### Predict NBA player performance with advanced AI models — powered by XGBoost.")
with col2:
    try:
        url = "https://assets4.lottiefiles.com/packages/lf20_x62chJ.json"
        lottie = requests.get(url).json()
        st_lottie(lottie, height=150, key="nba_lottie")
    except Exception:
        pass

st.divider()

# ======================================================
# 🎯 PLAYER INPUT SECTION
# ======================================================

player_name = st.text_input("Enter player name", placeholder="e.g. Luka Doncic")

if st.button("Run Predictions"):
    if not player_name:
        st.warning("Please enter a valid player name to continue.")
    else:
        with st.spinner("⚙️ Training model and generating predictions..."):
            try:
                player_id = get_player_id(player_name)
                context = get_player_context(player_name)
                if not context:
                    st.error("Unable to retrieve player data.")
                else:
                    df = build_feature_dataset(player_id)
                    if df is None or df.empty:
                        st.error("No sufficient data found for this player.")
                    else:
                        results = train_xgboost_models(df)
                        preds = predict_next_game(df)

                        # ======================================================
                        # 📊 RESULTS DISPLAY
                        # ======================================================
                        st.subheader(f"📈 Predicted Next Game Stats for {player_name}")

                        c1, c2, c3 = st.columns(3)
                        c1.metric("Points", f"{preds.get('PTS', 0):.1f}")
                        c2.metric("Rebounds", f"{preds.get('REB', 0):.1f}")
                        c3.metric("Assists", f"{preds.get('AST', 0):.1f}")
                        style_metric_cards(border_color="white", border_left_color="#00FFFF")

                        # Model metrics
                        st.divider()
                        st.markdown("### 🧩 Model Evaluation Metrics")
                        metrics_df = sanitize_dataframe_for_streamlit(pd.DataFrame(results).T)
                        st.dataframe(metrics_df, width="stretch")

                        # Visuals
                        st.divider()
                        st.markdown("### 🔮 Recent Performance Trend")
                        df_vis = df.tail(10).copy()
                        df_vis = sanitize_dataframe_for_streamlit(df_vis)
                        fig = px.line(
                            df_vis,
                            x="GAME_DATE",
                            y=["PTS", "REB", "AST"],
                            markers=True,
                            title=f"Recent 10 Games — {player_name}",
                            template="plotly_dark"
                        )
                        st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Error during prediction: {e}")

else:
    st.info("👆 Enter a player name above and click **Run Predictions** to start.")
