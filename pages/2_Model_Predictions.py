import streamlit as st
import pandas as pd
from utils.data_loader import get_player_id
from utils.feature_engineer import build_feature_dataset
from utils.model_utils import train_xgboost_models, predict_next_game
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_lottie import st_lottie
import requests

st.set_page_config(page_title="AI Predictions", page_icon="🤖", layout="wide")

def load_custom_css():
    with open("assets/theme.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
load_custom_css()

def load_lottieurl(url):
    r = requests.get(url)
    return r.json() if r.status_code == 200 else None

st.title("🤖 AI Model Predictions")
st.caption("Predict next game performance using advanced XGBoost models")
st_lottie(load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_q5pk6p1k.json"), height=140)

player = st.text_input("Enter player name", "Luka Doncic")

if st.button("Run Predictions"):
    with st.spinner("Training model and generating predictions..."):
        try:
            player_id = get_player_id(player)
            df = build_feature_dataset(player_id)

            results = train_xgboost_models(df)
            preds = predict_next_game(df)

            st.subheader("🔮 Next Game Projections")
            c1, c2, c3 = st.columns(3)
            c1.metric("Points", f"{preds.get('PTS', 0):.1f}")
            c2.metric("Rebounds", f"{preds.get('REB', 0):.1f}")
            c3.metric("Assists", f"{preds.get('AST', 0):.1f}")

            style_metric_cards(
                background_color="#1c1c1e",
                border_left_color="#f97316",
                border_color="#facc15",
                box_shadow=True,
            )

            st.subheader("📊 Model Evaluation Summary")
            st.dataframe(pd.DataFrame(results).T, use_container_width=True)
        except Exception as e:
            st.error(f"Error during prediction: {e}")
