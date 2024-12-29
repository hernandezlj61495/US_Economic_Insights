import os
import sqlite3
import subprocess
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from textblob import TextBlob
from pathlib import Path
from reportlab.pdfgen import canvas
import requests

# Ensure database path
db_path = Path("economic_data.db")

# Database Handling
if not db_path.exists():
    st.warning("Database file 'economic_data.db' is missing. Attempting to generate it...")
    try:
        subprocess.run(["python3", "data_processing.py"], check=True)
        st.success("Database successfully generated!")
    except Exception as e:
        st.error(f"Failed to generate the database: {e}")
        st.stop()

# Load data
def load_data():
    try:
        conn = sqlite3.connect(db_path)
        data = pd.read_sql("SELECT * FROM economic_data", conn)
        conn.close()
        return data
    except Exception as e:
        st.error(f"Failed to load data from database: {e}")
        st.stop()

# Streamlit App
st.set_page_config(page_title="US Economic Insights Dashboard", page_icon="📊", layout="wide")
st.title("📊 US Economic Insights Dashboard")

data = load_data()

# Visualizations
st.subheader("Clustered Economic Phases")
fig1 = px.scatter(data, x="GDP YoY Growth (%)", y="Unemployment Rate (%)", color="Economic Phase Name")
st.plotly_chart(fig1)

st.subheader("Trends Over Time")
fig2 = px.line(data, x="year", y=["GDP YoY Growth (%)", "Inflation Rate (%)", "Unemployment Rate (%)"])
st.plotly_chart(fig2)

# Forecasting
st.subheader("Forecasting")
indicator = st.selectbox("Select Indicator", ["GDP YoY Growth (%)", "Inflation Rate (%)", "Unemployment Rate (%)"])
if st.button("Prophet Forecast"):
    try:
        df = data.rename(columns={"year": "ds", indicator: "y"})
        model = Prophet()
        model.fit(df[["ds", "y"]])
        future = model.make_future_dataframe(periods=10)
        forecast = model.predict(future)
        fig3 = px.line(forecast, x="ds", y=["yhat", "yhat_lower", "yhat_upper"])
        st.plotly_chart(fig3)
    except Exception as e:
        st.error(f"Error: {e}")

# Monte Carlo Simulation
st.subheader("Monte Carlo Simulation")
initial = st.number_input("Initial Investment", value=1000)
growth = st.number_input("Growth Rate (%)", value=5.0)
years = st.slider("Years", 1, 50, 10)
results = [initial * (1 + np.random.normal(growth / 100, 0.02)) ** years for _ in range(1000)]
st.write(f"Median Value: ${np.median(results):,.2f}")
st.plotly_chart(px.histogram(results))

# News Sentiment
st.subheader("Live Economic News Sentiment")
api_key = "c4cda9e665ab468c8fbbc59df598fca3"
try:
    url = f"https://newsapi.org/v2/everything?q=economy&apiKey={api_key}"
    articles = requests.get(url).json().get("articles", [])
    for article in articles[:10]:
        sentiment = TextBlob(article["title"]).sentiment.polarity
        st.write(f"Title: {article['title']}")
        st.write(f"Sentiment: {'Positive' if sentiment > 0 else 'Negative' if sentiment < 0 else 'Neutral'}")
except Exception as e:
    st.error(f"News API error: {e}")

# Generate PDF
st.subheader("Generate Report")
if st.button("Download Report"):
    try:
        c = canvas.Canvas("report.pdf")
        c.drawString(100, 750, "Economic Report")
        c.save()
        with open("report.pdf", "rb") as pdf:
            st.download_button("Download Report", pdf)
    except Exception as e:
        st.error(f"PDF generation error: {e}")
