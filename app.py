import streamlit as st
import pandas as pd
import pickle

model = pickle.load(open("model/house_model.pkl", "rb"))

st.title("🏠 House Price Prediction")

bhk = st.number_input("Number of BHK", 1, 10)
area = st.number_input("Area (sq ft)", 200, 5000)
road_touch = st.selectbox("Road Touch?", ["Yes", "No"])
location = st.text_input("Location")

# Convert inputs
road_touch_val = 1 if road_touch == "Yes" else 0
data = pd.DataFrame([[bhk, area, road_touch_val]], columns=["BHK", "area", "road_touch"])

if st.button("Predict"):
    price = model.predict(data)[0]
    st.success(f"Estimated Price: ₹ {round(price, 2)}")
