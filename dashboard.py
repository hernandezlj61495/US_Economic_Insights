import os
import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import numpy as np
from prophet import Prophet
from reportlab.pdfgen import canvas
from textblob import TextBlob
import requests

# NewsAPI key
NEWSAPI_KEY = "c4cda9e665ab468c8fbbc59df598fca3"  # Replace with environment variable later if needed

# Load data from SQLite database
def load_data():
    try:
        if not os.path.exists('economic_data.db'):
            st.error("Database file 'economic_data.db' is missing. Please run `data_processing.py` to generate it.")
            st.stop()

        conn = sqlite3.connect('./economic_data.db')
        data = pd.read_sql('SELECT * FROM economic_data', conn)
        conn.close()
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# Forecasting Function
def forecast_indicator(data, column_name, periods=10):
    df = data[['year', column_name]].dropna()
    df.rename(columns={'year': 'ds', column_name: 'y'}, inplace=True)
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=periods, freq='Y')
    forecast = model.predict(future)
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

# Monte Carlo Simulation for Multi-Asset
def monte_carlo_multi_asset(initial_investment, asset_allocations, years, num_simulations=1000):
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
    c = canvas.Canvas(filename)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, 800, "US Economic Insights Report")

    c.setFont("Helvetica", 12)
    c.drawString(100, 780, "Key Economic Indicators:")

    gdp_growth_mean = data['GDP YoY Growth (%)'].mean()
    inflation_mean = data['Inflation Rate (%)'].mean()
    unemployment_mean = data['Unemployment Rate (%)'].mean()

    c.drawString(100, 760, f"- Average GDP Growth: {gdp_growth_mean:.2f}%")
    c.drawString(100, 740, f"- Average Inflation Rate: {inflation_mean:.2f}%")
    c.drawString(100, 720, f"- Average Unemployment Rate: {unemployment_mean:.2f}%")

    c.drawString(100, 700, "Clustering Summary:")
    cluster_counts = data['Economic Phase Name'].value_counts().to_dict()
    for cluster, count in cluster_counts.items():
        c.drawString(120, 680 - (20 * list(cluster_counts.keys()).index(cluster)), f"- {cluster}: {count} periods")

    if monte_carlo_results:
        median_value = np.median(monte_carlo_results)
        c.drawString(100, 580, "Monte Carlo Simulation Results:")
        c.drawString(120, 560, f"- Median Portfolio Value: ${median_value:,.2f}")
        c.drawString(120, 540, f"- 10th Percentile: ${np.percentile(monte_carlo_results, 10):,.2f}")
        c.drawString(120, 520, f"- 90th Percentile: ${np.percentile(monte_carlo_results, 90):,.2f}")

    c.setFont("Helvetica", 10)
    c.drawString(100, 480, "Report generated automatically by the US Economic Insights Dashboard.")
    c.save()

# Fetch Live News
def fetch_news(api_key, query='economy', max_results=5):
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

if economic_data.empty:
    st.error("No data available. Please populate the database by running `data_processing.py`.")
    st.stop()

# Clustered Economic Phases
st.subheader('Clustered Economic Phases')
fig = px.scatter(
    economic_data,
    x='GDP YoY Growth (%)',
    y='Unemployment Rate (%)',
    color='Economic Phase Name',
    title='Clustered Economic Phases with Labels'
)
st.plotly_chart(fig)

# Forecasting Section
st.subheader("Forecasting Economic Indicators")
indicator = st.selectbox(
    "Select an Indicator to Forecast",
    ['GDP YoY Growth (%)', 'Inflation Rate (%)', 'Unemployment Rate (%)']
)

if indicator:
    forecast = forecast_indicator(economic_data, indicator, periods=10)
    st.write(f"Forecast for {indicator}")
    fig = px.line(
        forecast,
        x='ds',
        y='yhat',
        title=f'Forecast for {indicator}',
        labels={'ds': 'Year', 'yhat': f'Forecasted {indicator}'}
    )
    fig.add_scatter(
        x=forecast['ds'], 
        y=forecast['yhat_lower'], 
        mode='lines', 
        name='Lower Bound', 
        line=dict(dash='dot')
    )
    fig.add_scatter(
        x=forecast['ds'], 
        y=forecast['yhat_upper'], 
        mode='lines', 
        name='Upper Bound', 
        line=dict(dash='dot')
    )
    st.plotly_chart(fig)

# Monte Carlo Simulation Section
st.subheader("Multi-Asset Portfolio Growth Simulator")
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
