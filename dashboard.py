import os
import sqlite3
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from prophet import Prophet
from textblob import TextBlob
from reportlab.pdfgen import canvas
import requests

# Functions to dynamically create and handle the database
def fetch_data():
    years = list(range(2000, 2023))  # Dummy data for 23 years
    gdp_growth = [-1, 2, 3] * (len(years) // 3) + [-1] * (len(years) % 3)
    inflation = [3, 4, 5] * (len(years) // 3) + [3] * (len(years) % 3)
    unemployment = [5, 6, 7] * (len(years) // 3) + [5] * (len(years) % 3)

    data = {
        "year": years,
        "gdp_growth": gdp_growth[:len(years)],
        "inflation": inflation[:len(years)],
        "unemployment": unemployment[:len(years)],
    }
    return pd.DataFrame(data)

def process_data(df):
    df["rolling_avg_gdp"] = df["gdp_growth"].rolling(window=3).mean()
    features = df[["gdp_growth", "inflation", "unemployment"]].dropna()
    kmeans = KMeans(n_clusters=3, random_state=42)
    df["economic_phase"] = kmeans.fit_predict(features)
    df["economic_phase_name"] = df["economic_phase"].map({0: "Recession", 1: "Growth", 2: "Stagflation"})
    return df

def save_to_db(df):
    conn = sqlite3.connect("economic_data.db")
    df.to_sql("economic_data", conn, if_exists="replace", index=False)
    conn.close()

def ensure_database():
    db_path = "economic_data.db"
    if not os.path.exists(db_path):
        raw_data = fetch_data()
        processed_data = process_data(raw_data)
        save_to_db(processed_data)

@st.cache_data
def load_data():
    ensure_database()
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

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Visualizations", "Forecasting", "Simulations", "News Sentiment", "Generate Report"
])

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

with tab4:
    st.subheader("Live News Sentiment Analysis")

    # Description of the feature
    st.write("""
        This feature retrieves the latest news articles about the economy and analyzes the sentiment 
        of their headlines. Positive, negative, or neutral sentiments provide insights into the current 
        perception of economic events in the media.
    """)

    # Using your NewsAPI key
    api_key = "c4cda9e665ab468c8fbbc59df598fca3"
    url = f"https://newsapi.org/v2/everything?q=economy&language=en&sortBy=publishedAt&apiKey={api_key}"

    try:
        # Fetching the news articles
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for HTTP issues
        articles = response.json().get("articles", [])

        if articles:
            for article in articles[:5]:  # Display the top 5 articles
                title = article.get("title", "No title available")
                description = article.get("description", "No description available")
                url = article.get("url", "")

                # Analyze sentiment of the article's title
                sentiment_score = TextBlob(title).sentiment.polarity
                sentiment_label = "Positive" if sentiment_score > 0 else "Negative" if sentiment_score < 0 else "Neutral"

                # Display the article with its sentiment
                st.write(f"**[{title}]({url})** ({sentiment_label})")
                st.write(f"*{description}*")
                st.write("---")  # Divider
        else:
            st.write("No news articles found for the query.")
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch news: {e}")


with tab5:
    st.subheader("Generate Economic Report")
    if st.button("Download Report"):
        try:
            generate_pdf(data)
            with open("economic_report.pdf", "rb") as pdf:
                st.download_button("Download Report", pdf, file_name="economic_report.pdf")
        except Exception as e:
            st.error(f"PDF generation failed: {e}")
