import sqlite3
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from prophet import Prophet
from textblob import TextBlob
from reportlab.pdfgen import canvas
import requests

@st.cache_data
def load_data():
    conn = sqlite3.connect("economic_data.db")
    data = pd.read_sql("SELECT * FROM economic_data", conn)
    conn.close()
    return data

def generate_pdf(data):
    c = canvas.Canvas("economic_report.pdf")
    c.drawString(100, 750, "US Economic Insights Report")
    avg_gdp = data['gdp_growth'].mean()
    avg_inflation = data['inflation'].mean()
    avg_unemployment = data['unemployment'].mean()
    c.drawString(100, 730, f"Average GDP Growth: {avg_gdp:.2f}%")
    c.drawString(100, 710, f"Average Inflation Rate: {avg_inflation:.2f}%")
    c.drawString(100, 690, f"Average Unemployment Rate: {avg_unemployment:.2f}%")
    c.save()

st.title("US Economic Insights Dashboard")
data = load_data()

# Tabs for navigation
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Visualizations", "Forecasting", "Simulations", "News Sentiment", "Generate Report"
])

# Visualizations
with tab1:
    st.subheader("Economic Trends")
    fig = px.line(data, x="year", y=["gdp_growth", "inflation", "unemployment"],
                  labels={"value": "Percentage", "variable": "Indicator"},
                  title="Economic Indicators Over Time")
    st.plotly_chart(fig)

    st.subheader("Clustered Economic Phases")
    if "economic_phase_name" in data.columns:
        fig2 = px.scatter(data, x="gdp_growth", y="unemployment",
                          color="economic_phase_name", title="Economic Phases")
        st.plotly_chart(fig2)

# Forecasting
with tab2:
    st.subheader("Forecasting")
    indicator = st.selectbox("Select Indicator", ["gdp_growth", "inflation", "unemployment"])
    if st.button("Run Forecast"):
        try:
            df = data.rename(columns={"year": "ds", indicator: "y"})
            model = Prophet()
            model.fit(df[["ds", "y"]])
            future = model.make_future_dataframe(periods=10)
            forecast = model.predict(future)
            fig3 = px.line(forecast, x="ds", y=["yhat", "yhat_lower", "yhat_upper"],
                           labels={"ds": "Year", "value": "Forecast"},
                           title=f"Forecast for {indicator}")
            st.plotly_chart(fig3)
        except Exception as e:
            st.error(f"Forecasting failed: {e}")

# Simulations
with tab3:
    st.subheader("Monte Carlo Simulation")
    initial_investment = st.number_input("Initial Investment ($)", value=1000.0)
    growth_rate = st.slider("Expected Annual Growth Rate (%)", 0.0, 20.0, 5.0)
    years = st.slider("Number of Years", 1, 50, 10)
    simulations = st.number_input("Number of Simulations", value=1000, step=100)

    if st.button("Run Simulation"):
        results = []
        for _ in range(simulations):
            yearly_growth = np.random.normal(growth_rate / 100, 0.02, years)
            portfolio_value = initial_investment * np.cumprod(1 + yearly_growth)
            results.append(portfolio_value[-1])

        st.write(f"Median Portfolio Value: ${np.median(results):,.2f}")
        fig4 = px.histogram(results, nbins=50, title="Monte Carlo Simulation Results")
        fig4.update_layout(xaxis_title="Portfolio Value ($)", yaxis_title="Frequency")
        st.plotly_chart(fig4)

# News Sentiment
with tab4:
    st.subheader("Live News Sentiment Analysis")
    api_key = "your_news_api_key"  # Replace with your NewsAPI key
    try:
        url = f"https://newsapi.org/v2/everything?q=economy&apiKey={api_key}"
        articles = requests.get(url).json().get("articles", [])
        for article in articles[:5]:
            sentiment = TextBlob(article["title"]).sentiment.polarity
            sentiment_label = "Positive" if sentiment > 0 else "Negative" if sentiment < 0 else "Neutral"
            st.write(f"**{article['title']}** ({sentiment_label})")
            st.write(article["description"])
    except Exception as e:
        st.error(f"Error fetching news: {e}")

# Generate PDF Report
with tab5:
    st.subheader("Generate Economic Report")
    if st.button("Download Report"):
        try:
            generate_pdf(data)
            with open("economic_report.pdf", "rb") as pdf:
                st.download_button("Download Report", pdf, file_name="economic_report.pdf")
        except Exception as e:
            st.error(f"PDF generation failed: {e}")
