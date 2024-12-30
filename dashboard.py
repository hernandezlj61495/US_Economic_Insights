import os
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from prophet import Prophet
from textblob import TextBlob
from reportlab.pdfgen import canvas
import requests
import matplotlib.pyplot as plt

# Suppress sklearn FutureWarnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

# Fetch Data Function
def fetch_data(live_data=False):
    current_year = pd.Timestamp.now().year
    if live_data:
        try:
            api_url = "https://api.worldbank.org/v2/country/USA/indicator/NY.GDP.MKTP.CD?format=json"
            response = requests.get(api_url)
            response.raise_for_status()
            data = response.json()
            if len(data) > 1 and "value" in data[1][0]:
                df = pd.DataFrame(data[1])
                df = df[["date", "value"]].rename(columns={"date": "year", "value": "gdp_growth"})
                df["year"] = df["year"].astype(int)
                df["inflation"] = np.random.uniform(1, 3, len(df))  # Placeholder for inflation
                df["unemployment"] = np.random.uniform(3, 7, len(df))  # Placeholder for unemployment
                return df[df["year"] >= current_year - 10]
            else:
                st.warning("Live data unavailable. Using static fallback.")
        except Exception as e:
            st.error(f"Error fetching live data: {e}")

    # Fallback data for the last 10 years
    years = list(range(current_year - 10, current_year + 1))
    gdp_growth = np.random.uniform(1, 5, len(years))
    inflation = np.random.uniform(2, 4, len(years))
    unemployment = np.random.uniform(3, 8, len(years))
    data = {
        "year": years,
        "gdp_growth": gdp_growth,
        "inflation": inflation,
        "unemployment": unemployment,
    }
    return pd.DataFrame(data)

# Process Data Function
def process_data(df):
    df["rolling_avg_gdp"] = df["gdp_growth"].rolling(window=3).mean()
    features = df[["gdp_growth", "inflation", "unemployment"]].dropna()
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df["economic_phase"] = kmeans.fit_predict(features)
    df["economic_phase_name"] = df["economic_phase"].map({0: "Recession", 1: "Growth", 2: "Stagflation"})
    return df

# Generate PDF Report
def generate_pdf(data):
    current_year = pd.Timestamp.now().year
    latest_data = data[data["year"] == current_year]

    c = canvas.Canvas("economic_report.pdf")
    c.drawString(100, 750, "US Economic Insights Report")
    c.drawString(100, 730, f"Last Updated: {pd.Timestamp.now().strftime('%Y-%m-%d')}")

    if not latest_data.empty:
        gdp = latest_data["gdp_growth"].values[0]
        inflation = latest_data["inflation"].values[0]
        unemployment = latest_data["unemployment"].values[0]
        c.drawString(100, 710, f"Current Year ({current_year}):")
        c.drawString(100, 690, f"- GDP Growth: {gdp:.2f}%")
        c.drawString(100, 670, f"- Inflation: {inflation:.2f}%")
        c.drawString(100, 650, f"- Unemployment: {unemployment:.2f}%")
    else:
        c.drawString(100, 710, f"No Data Available for {current_year}.")

    c.drawString(100, 630, "Key Insights:")
    c.drawString(100, 610, "- During high inflation, secure assets perform well.")
    c.drawString(100, 590, "- Growth phases are ideal for investments.")

    fig, ax = plt.subplots()
    data.plot(x="year", y=["gdp_growth", "inflation", "unemployment"], ax=ax)
    ax.set_title("Economic Trends Over Time")
    plt.savefig("trends.png")
    c.drawImage("trends.png", 100, 400, width=400, height=200)

    c.save()

# Live Sentiment Analysis Function
def analyze_sentiment():
    api_key = "c4cda9e665ab468c8fbbc59df598fca3"
    url = f"https://newsapi.org/v2/everything?q=economy&language=en&apiKey={api_key}"

    try:
        response = requests.get(url)
        response.raise_for_status()
        articles = response.json().get("articles", [])
        sentiments = []
        for article in articles[:5]:
            title = article.get("title", "No title available")
            sentiment_score = TextBlob(title).sentiment.polarity
            sentiments.append(sentiment_score)
        avg_sentiment = np.mean(sentiments)
        return avg_sentiment
    except Exception as e:
        st.error(f"Error fetching sentiment: {e}")
        return 0  # Neutral fallback sentiment

# Main Dashboard Implementation
live_data = st.sidebar.checkbox("Use Live Data", value=False)
data = fetch_data(live_data=live_data)
data = process_data(data)

tabs = st.tabs(["Overview", "Visualizations", "Forecasting", "Simulations", "Live Sentiment", "Generate Report"])

# Overview Tab
with tabs[0]:
    st.image("banner.jpg", use_container_width=True)
    st.markdown("""
    ### Welcome to the US Economic Insights Dashboard
    This dashboard provides:
    - **Economic Trends**: Visualize key metrics like GDP, inflation, and unemployment.
    - **Forecasting Tools**: Predict economic indicators to guide decision-making.
    - **Monte Carlo Simulations**: Assess investment risks and returns.
    - **Live Sentiment Analysis**: Stay updated on market sentiment.
    """)

    st.markdown("### Economic Insights Clock")
    st.write("Live updates on key economic metrics and insights:")

    # Calculate Economic Health Score
    current_data = data.iloc[-1]
    gdp_growth = current_data["gdp_growth"]
    inflation = current_data["inflation"]
    unemployment = current_data["unemployment"]
    avg_sentiment = analyze_sentiment()

    gdp_score = np.clip((gdp_growth - 1) / (5 - 1) * 100, 0, 100)
    inflation_score = np.clip((4 - inflation) / (4 - 2) * 100, 0, 100)
    unemployment_score = np.clip((8 - unemployment) / (8 - 3) * 100, 0, 100)
    sentiment_score = np.clip((avg_sentiment + 1) / 2 * 100, 0, 100)

    economic_health_score = round(
        0.4 * gdp_score + 0.3 * inflation_score + 0.2 * unemployment_score + 0.1 * sentiment_score, 2
    )

    st.metric("GDP Growth", f"{gdp_growth:.2f}%")
    st.metric("Inflation", f"{inflation:.2f}%")
    st.metric("Unemployment", f"{unemployment:.2f}%")
    st.metric("Economic Health Score", f"{economic_health_score}/100")

# Visualization Tab
with tabs[1]:
    st.subheader("Economic Trends Over Time")
    fig = px.line(data, x="year", y=["gdp_growth", "inflation", "unemployment"],
                  labels={"value": "Percentage", "variable": "Indicator"},
                  title="Economic Indicators Over Time")
    st.plotly_chart(fig)

# Forecasting Tab
with tabs[2]:
    st.subheader("Forecasting")
    indicator = st.selectbox("Select Indicator", ["gdp_growth", "inflation", "unemployment"])
    forecast_horizon = st.slider("Forecast Horizon (Years)", 1, 20, 10)
    if st.button("Run Forecast"):
        df = data.rename(columns={"year": "ds", indicator: "y"})
        model = Prophet()
        model.fit(df[["ds", "y"]])
        future = model.make_future_dataframe(periods=forecast_horizon)
        forecast = model.predict(future)
        fig = px.line(forecast, x="ds", y=["yhat", "yhat_lower", "yhat_upper"],
                      labels={"ds": "Year", "value": "Forecast"},
                      title=f"Forecast for {indicator.capitalize()}")
        st.plotly_chart(fig)

# Simulations Tab
with tabs[3]:
    st.subheader("Monte Carlo Simulation")
    initial_investment = st.number_input("Initial Investment ($)", value=1000.0)
    growth_rate = st.slider("Annual Growth Rate (%)", 0.0, 20.0, 5.0)
    years = st.slider("Investment Period (Years)", 1, 30, 10)
    if st.button("Run Simulation"):
        results = [initial_investment * np.prod(1 + np.random.normal(growth_rate / 100, 0.02, years))
                   for _ in range(1000)]
        fig = px.histogram(results, nbins=50, title="Monte Carlo Simulation Results")
        st.plotly_chart(fig)

# Live Sentiment Tab
with tabs[4]:
    avg_sentiment = analyze_sentiment()
    st.metric("Overall Sentiment Score", avg_sentiment)

# Generate Report Tab
with tabs[5]:
    st.subheader("Generate Report")
    if st.button("Download Report"):
        generate_pdf(data)
        with open("economic_report.pdf", "rb") as pdf:
            st.download_button("Download Report", pdf, file_name="economic_report.pdf")
