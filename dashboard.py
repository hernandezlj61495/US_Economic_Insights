import os
import subprocess
import sqlite3
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from textblob import TextBlob
from reportlab.pdfgen import canvas
import requests

# NewsAPI Key
NEWSAPI_KEY = "c4cda9e665ab468c8fbbc59df598fca3"

# Check if the database exists, and if not, generate it
if not os.path.exists('economic_data.db'):
    st.warning("Database file 'economic_data.db' is missing. Generating it now...")
    try:
        # Run data_processing.py to generate the database
        result = subprocess.run(
            ["python3", "data_processing.py"], 
            check=True, 
            capture_output=True, 
            text=True
        )
        st.success("Database generated successfully!")
    except subprocess.CalledProcessError as e:
        st.error(f"Failed to generate the database. Error: {e.stderr}")
        st.stop()

# Load data from SQLite database
def load_data():
    try:
        conn = sqlite3.connect('economic_data.db')
        data = pd.read_sql('SELECT * FROM economic_data', conn)
        conn.close()
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

# Forecasting Functions
def forecast_indicator(data, column_name, periods=10):
    """Forecast future trends using Prophet."""
    df = data[['year', column_name]].dropna()
    df.rename(columns={'year': 'ds', column_name: 'y'}, inplace=True)
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=periods, freq='Y')
    forecast = model.predict(future)
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

def arima_forecast(data, column_name, periods=10):
    """Forecast future trends using ARIMA."""
    df = data[['year', column_name]].dropna()
    df.set_index('year', inplace=True)
    model = ARIMA(df[column_name], order=(1, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=periods)
    return forecast

# Monte Carlo Simulation for Multi-Asset
def monte_carlo_multi_asset(initial_investment, asset_allocations, years, num_simulations=1000):
    """Simulate portfolio growth under different conditions."""
    results = []
    for _ in range(num_simulations):
        portfolio_value = initial_investment
        for year in range(years):
            for asset, (allocation, growth_rate) in asset_allocations.items():
                portfolio_value += (portfolio_value * allocation) * np.random.normal(growth_rate / 100, 0.02)
        results.append(portfolio_value)
    return results

# PDF Report Generator
def generate_report(filename, data, monte_carlo_results=None):
    """Generate a professional PDF report."""
    c = canvas.Canvas(filename)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, 750, "US Economic Insights Report")

    c.setFont("Helvetica", 12)
    c.drawString(100, 730, "Key Economic Indicators:")

    # Add summary statistics
    gdp_growth_mean = data['GDP YoY Growth (%)'].mean()
    inflation_mean = data['Inflation Rate (%)'].mean()
    unemployment_mean = data['Unemployment Rate (%)'].mean()

    c.drawString(100, 710, f"- Average GDP Growth: {gdp_growth_mean:.2f}%")
    c.drawString(100, 690, f"- Average Inflation Rate: {inflation_mean:.2f}%")
    c.drawString(100, 670, f"- Average Unemployment Rate: {unemployment_mean:.2f}%")

    # Add Monte Carlo results
    if monte_carlo_results:
        median_value = np.median(monte_carlo_results)
        c.drawString(100, 650, "Monte Carlo Simulation Results:")
        c.drawString(120, 630, f"- Median Portfolio Value: ${median_value:,.2f}")
        c.drawString(120, 610, f"- 10th Percentile: ${np.percentile(monte_carlo_results, 10):,.2f}")
        c.drawString(120, 590, f"- 90th Percentile: ${np.percentile(monte_carlo_results, 90):,.2f}")

    # Final note
    c.setFont("Helvetica", 10)
    c.drawString(100, 570, "Report generated automatically by the US Economic Insights Dashboard.")
    c.save()

# Fetch Live News
def fetch_news(api_key, query='economy', max_results=5):
    """Fetch live economic news headlines."""
    url = f"https://newsapi.org/v2/everything?q={query}&apiKey={api_key}&pageSize={max_results}"
    response = requests.get(url)
    if response.status_code == 200:
        articles = response.json().get('articles', [])
        return [article['title'] for article in articles]
    else:
        st.error("Failed to fetch news. Check your API key or query.")
        return []

# Main App
st.title('US Economic Insights Dashboard')
economic_data = load_data()

# Clustered Economic Phases
st.subheader('Clustered Economic Phases')
fig = px.scatter(
    economic_data,
    x='GDP YoY Growth (%)',
    y='Unemployment Rate (%)',
    color='Economic Phase Name',
    title='Clustered Economic Phases'
)
st.plotly_chart(fig)

# Forecasting Section
st.subheader("Forecasting Economic Indicators")
indicator = st.selectbox(
    "Select an Indicator to Forecast",
    ['GDP YoY Growth (%)', 'Inflation Rate (%)', 'Unemployment Rate (%)']
)

if indicator:
    st.write(f"ARIMA Forecast for {indicator}:")
    arima_values = arima_forecast(economic_data, indicator)
    st.write(arima_values)

    st.write(f"Prophet Forecast for {indicator}:")
    prophet_values = forecast_indicator(economic_data, indicator)
    st.write(prophet_values)

# Monte Carlo Simulation Section
st.subheader("Portfolio Growth Simulator")
initial_investment = st.number_input('Initial Investment ($)', min_value=1000, step=100)
stocks_allocation = st.slider("Stocks Allocation (%)", 0, 100, 60)
bonds_allocation = st.slider("Bonds Allocation (%)", 0, 100, 40)
years = st.slider('Investment Period (Years)', 1, 50, 10)

if stocks_allocation + bonds_allocation > 100:
    st.error("Total allocation cannot exceed 100%")
else:
    asset_allocations = {
        "Stocks": (stocks_allocation / 100, 8),
        "Bonds": (bonds_allocation / 100, 3),
    }
    if st.button("Run Monte Carlo Simulation"):
        results = monte_carlo_multi_asset(initial_investment, asset_allocations, years)
        st.write(f"Median Portfolio Value: **${np.median(results):,.2f}**")
        fig = px.histogram(results, nbins=50, title="Portfolio Value Distribution")
        st.plotly_chart(fig)

# Live Sentiment Analysis
st.subheader("Live Economic News Sentiment")
headlines = fetch_news(NEWSAPI_KEY)
sentiments = [TextBlob(headline).sentiment.polarity for headline in headlines]
for headline, sentiment in zip(headlines, sentiments):
    sentiment_label = 'Positive' if sentiment > 0 else 'Negative' if sentiment < 0 else 'Neutral'
    st.write(f'"{headline}" - Sentiment: {sentiment_label}')

# PDF Report Generation
st.subheader("Download Economic Report")
if st.button("Generate Report"):
    monte_carlo_results = monte_carlo_multi_asset(initial_investment, asset_allocations, years) if initial_investment > 0 else None
    generate_report("economic_report.pdf", economic_data, monte_carlo_results)
    with open("economic_report.pdf", "rb") as f:
        st.download_button("Download Report", f, file_name="economic_report.pdf")
