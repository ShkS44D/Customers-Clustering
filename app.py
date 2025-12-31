import streamlit as st
import joblib
import numpy as np

model = joblib.load("kmeans_model.pkl")

st.title("Customer Segmentation App")

income = st.slider("Annual Income (k$)", 10, 150)
score = st.slider("Spending Score (1-100)", 1, 100)

prediction = model.predict([[income, score]])

st.success(f"Customer belongs to Cluster: {prediction[0]}")
