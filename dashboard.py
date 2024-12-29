import os
from pathlib import Path
import sqlite3
import subprocess
import pandas as pd
import streamlit as st
import plotly.express as px
from textblob import TextBlob
import numpy as np
from reportlab.pdfgen import canvas
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA

# Set database path
db_path = Path(os.getenv('STREAMLIT_TEMP_DIR', '.')) / "economic_data.db"

# Ensure the database exists
if not db_path.exists():
    st.warning("Database file is missing. Generating it now...")
    try:
        result = subprocess.run(
            ["python3", "data_processing.py"],
            check=True,
            capture_output=True,
            text=True
        )
        st.success("Database generated successfully!")
    except subprocess.CalledProcessError as e:
        st.error(f"Failed to generate the database. Error details: {e.stderr}")
        st.stop()

# Load data from the database
def load_data():
    """Load data from SQLite database."""
    try:
        conn = sqlite3.connect(db_path)
        data = pd.read_sql("SELECT * FROM economic_data", conn)
        conn.close()
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

# Streamlit app
st.set_page_config(
    page_title="US Economic Insights Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 US Economic Insights Dashboard")

# Load and display data
data = load_data()

# --- Section 1: Clustered Economic Phases ---
st.markdown("## Clustered Economic Phases")
st.markdown("### Visualize economic phases based on GDP growth, unemployment rate, and inflation rate.")
fig1 = px.scatter(
    data,
    x="GDP YoY Growth (%)",
    y="Unemployment Rate (%)",
    color="Economic Phase Name",
    title="Clustered Economic Phases",
    labels={"GDP YoY Growth (%)": "GDP Growth (%)", "Unemployment Rate (%)": "Unemployment (%)"}
)
st.plotly_chart(fig1, use_container_width=True)

# --- Section 2: Economic Indicators Trends ---
st.markdown("## Economic Indicators Trends")
st.markdown("### Track trends for GDP growth, inflation, and unemployment over time.")
fig2 = px.line(
    data,
    x="year",
    y=["GDP YoY Growth (%)", "Inflation Rate (%)", "Unemployment Rate (%)"],
    title="Economic Indicators Over Time",
    labels={"value": "Percentage", "variable": "Indicator"}
)
st.plotly_chart(fig2, use_container_width=True)

# --- Section 3: Advanced Forecasting ---
st.markdown("## Advanced Forecasting")
indicator = st.selectbox("Select an Indicator to Forecast", ["GDP YoY Growth (%)", "Inflation Rate (%)", "Unemployment Rate (%)"])

# Prophet Model
if st.button("Generate Prophet Forecast"):
    try:
        df = data[["year", indicator]].rename(columns={"year": "ds", indicator: "y"})
        model = Prophet()
        model.fit(df)
        future = model.make_future_dataframe(periods=10)
        forecast = model.predict(future)

        st.markdown(f"### Prophet Forecast for {indicator}")
        fig3 = px.line(forecast, x="ds", y=["yhat", "yhat_lower", "yhat_upper"], title=f"Prophet Forecast for {indicator}")
        st.plotly_chart(fig3, use_container_width=True)
    except Exception as e:
        st.error(f"Error during Prophet forecasting: {e}")

# ARIMA Model
if st.button("Generate ARIMA Forecast"):
    try:
        arima_model = ARIMA(data[indicator], order=(1, 1, 1)).fit()
        forecast = arima_model.get_forecast(steps=10)
        forecast_df = forecast.summary_frame()

        st.markdown(f"### ARIMA Forecast for {indicator}")
        fig4 = px.line(forecast_df, x=forecast_df.index, y=["mean", "mean_ci_lower", "mean_ci_upper"], title=f"ARIMA Forecast for {indicator}")
        st.plotly_chart(fig4, use_container_width=True)
    except Exception as e:
        st.error(f"Error during ARIMA forecasting: {e}")

# --- Section 4: Monte Carlo Simulation ---
st.markdown("## Portfolio Growth Simulator")
initial_investment = st.number_input("Initial Investment ($)", min_value=100, value=1000, step=100)
annual_growth_rate = st.number_input("Annual Growth Rate (%)", min_value=0.0, value=7.0, step=0.1)
years = st.slider("Investment Duration (Years)", min_value=1, max_value=50, value=10)
market_condition = st.selectbox("Market Condition", ["Bullish", "Neutral", "Bearish"])

# Adjust simulation parameters based on market condition
scale = 0.02 if market_condition == "Neutral" else (0.03 if market_condition == "Bullish" else 0.01)
simulations = 1000
results = []
for _ in range(simulations):
    growth_rates = np.random.normal(loc=annual_growth_rate / 100, scale=scale, size=years)
    growth = initial_investment
    for rate in growth_rates:
        growth *= (1 + rate)
    results.append(growth)

st.markdown(f"### Projected Portfolio Value After {years} Years")
st.write(f"Median Value: ${np.median(results):,.2f}")
fig5 = px.histogram(results, nbins=50, title="Monte Carlo Simulation Results")
st.plotly_chart(fig5, use_container_width=True)

# --- Section 5: Live News Sentiment Analysis ---
st.markdown("## Live Economic News Sentiment")
api_key = st.text_input("Enter your NewsAPI key:")
if api_key:
    try:
        import requests
        url = f"https://newsapi.org/v2/everything?q=economy&apiKey={api_key}"
        response = requests.get(url)
        articles = response.json().get("articles", [])

        sentiments = []
        for article in articles[:10]:  # Limit to 10 articles for brevity
            sentiment = TextBlob(article["title"]).sentiment.polarity
            sentiments.append((article["title"], sentiment))

        for title, sentiment in sentiments:
            st.write(f"Title: {title}")
            st.write(f"Sentiment: {'Positive' if sentiment > 0 else 'Negative' if sentiment < 0 else 'Neutral'}")
    except Exception as e:
        st.error(f"Error fetching or analyzing news sentiment: {e}")

# --- Section 6: Professional PDF Report ---
st.markdown("## Generate Economic Report")
if st.button("Generate PDF Report"):
    try:
        temp_dir = Path(os.getenv('STREAMLIT_TEMP_DIR', '.'))
        pdf_path = temp_dir / "economic_report.pdf"

        # Generate the PDF
        c = canvas.Canvas(str(pdf_path))
        c.drawString(100, 750, "US Economic Insights Report")
        c.drawString(100, 700, f"Average GDP Growth: {data['GDP YoY Growth (%)'].mean():.2f}%")
        c.drawString(100, 675, f"Average Inflation Rate: {data['Inflation Rate (%)'].mean():.2f}%")
        c.drawString(100, 650, f"Average Unemployment Rate: {data['Unemployment Rate (%)'].mean():.2f}%")
        c.save()

        with open(pdf_path, "rb") as pdf_file:
            st.download_button(
                label="Download Economic Report",
                data=pdf_file,
                file_name="economic_report.pdf",
                mime="application/pdf"
            )
        st.success("Report generated! Use the button above to download.")
    except Exception as e:
        st.error(f"Error generating the PDF report: {e}")

# --- Section 7: Interactive Glossary ---
st.markdown("## Interactive Glossary")
with st.expander("What is GDP?"):
    st.write("Gross Domestic Product (GDP) is the total monetary value of all goods and services produced within a country's borders.")
with st.expander("What is Inflation?"):
    st.write("Inflation is the rate at which the general level of prices for goods and services rises, eroding purchasing power.")
with st.expander("What is a Recession?"):
    st.write("A recession is a period of economic decline, typically identified by a fall in GDP for two consecutive quarters.")
