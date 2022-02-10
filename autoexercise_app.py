import base64
import numpy as np
import pandas as pd
import imageio
import streamlit as st
import plotly.express as px
from utils import get_sequence_model, sequence_prediction, build_feature_extractor

# load sequence model
MAX_SEQ_LENGTH = 120
MODEL_PATH = "sequence_model/"  # change this line to saved model path
sequence_model = get_sequence_model(MODEL_PATH)

st.title("Exercise classification")
st.write("""Upload your video to predict which exercise you are doing.""")
uploaded_video = st.file_uploader("Choose an exercise video...", type=["mp4", "mov"])


def to_gif(frames, file_name="animation.gif"):
    """Convert frames to gif"""
    converted_images = frames.astype(np.uint8)
    imageio.mimsave(file_name, converted_images, fps=10)


@st.cache
def create_feature_extractor():
    feature_extractor = build_feature_extractor()
    return feature_extractor
feature_extractor = create_feature_extractor()


if uploaded_video is not None:
    vid = uploaded_video.name
    st.markdown(
        f"File name: {vid}",
        unsafe_allow_html=True,
    )  # display file name

    with open(vid, mode="wb") as f:
        f.write(uploaded_video.read())  # save video to disk

    frames, probabilities = sequence_prediction(vid, sequence_model, feature_extractor)
    to_gif(frames[:MAX_SEQ_LENGTH])  # save array to GIF

    # display GIF and write preidction output
    with open("animation.gif", "rb") as f:
        gif = base64.b64encode(f.read()).decode("utf-8")
    st.markdown(
        f'<img src="data:image/gif;base64,{gif}" alt="cat gif">',
        unsafe_allow_html=True,
    )
    st.write("**Prediction**:\n")
    prob_df = pd.DataFrame(
        list(probabilities.items()), columns=["Exercise", "Probability"]
    )
    pred_exercise = prob_df.sort_values(by="Probability", ascending=False)[
        "Exercise"
    ].iloc[0]
    st.write(f"You are doing {pred_exercise}")
    fig = px.bar(prob_df, x="Exercise", y="Probability", barmode="group", height=400)
    st.plotly_chart(fig)
