import streamlit as st
from tennis_predictor import predict_play_tennis


api_key = st.secrets["api"]["weather_key"]

st.set_page_config(page_title=" Should I play tennis?", layout="centered")
st.title("Should I play tennis?")

st.write("This app determines whether tennis can be played based on weather and court density forecast.")


city = st.text_input("City Name", value="Istanbul")


hour = st.slider("What time do you plan to play?", min_value=0, max_value=23, value=18)


if st.button("Predict"):
    with st.spinner("Making prediction..."):
        result = predict_play_tennis(city, hour, api_key)
    st.success(result)
