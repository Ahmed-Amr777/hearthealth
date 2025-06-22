import streamlit as st

st.title("Heart Disease Predictor")
age = st.slider("Age", 20, 80)
st.write("Your age is", age)
