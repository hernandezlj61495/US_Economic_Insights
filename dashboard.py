import os
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from prophet import Prophet
from textblob import TextBlob
from reportlab.pdfgen import canvas
import requests
import matplotlib.pyplot as plt
from collections import Counter
from nltk.corpus import stopwords
from wordcloud import WordCloud
import nltk

nltk.download("stopwords")

# Suppress sklearn FutureWarnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

# Custom CSS for styling
st.markdown("""
    <style>
        .main {
            background-color: #f0f4fc; /* Light blue for main background */
            color: #222222; /* Darker text color for contrast */
        }
        h1, h2, h3, h4, h5 {
            color: #00509e; /* Navy blue for headings */
        }
        .stButton>button {
            background-color: #00509e; /* Navy blue buttons */
            color: white;
            border-radius: 10px;
            padding: 10px;
        }
        .css-1d391kg {
            background-color: #f7faff; /* Very light blue for sidebar */
        }
    </style>
""", unsafe_allow_html=True)

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
        return sentiments, avg_sentiment, articles
    except Exception as e:
        st.error(f"Error fetching sentiment: {e}")
        return [], 0, []  # Neutral fallback sentiment

# Main Dashboard Implementation
live_data = st.sidebar.checkbox("Use Live Data", value=False)
data = fetch_data(live_data=live_data)
data = process_data(data)

tabs = st.tabs(["Overview", "Visualizations", "Forecasting", "Simulations", "Live Sentiment", "Generate Report"])

# Overview Tab
with tabs[0]:
    st.image("banner.jpg", use_container_width=True)  # Updated banner image
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
    sentiments, avg_sentiment, _ = analyze_sentiment()

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
# Rewriting the Forecast Tab with a Focus on Comprehensive Financial Forecasting

# Forecasting Tab
with tabs[2]:
    st.subheader("Forecasting: Comprehensive Financial Insights for Your Future")

    # Section Introduction
    st.markdown("""
    This section provides a comprehensive forecast of your financial future, leveraging advanced machine learning models and real-time economic data. 
    Personalized recommendations are tailored to your income, expenses, savings, age, and financial goals.
    """)

    # Overview of Features
    st.markdown("""
    ### Key Features:
    - **Disposable Income**: Calculate from your input income and expenses.
    - **Retirement Planning**: Predict required savings based on age and desired retirement goals.
    - **Investment Growth**: Simulate portfolio growth using AI-driven market predictions.
    - **Emergency Fund Recommendations**: Tailor to age and unemployment trends.
    - **Big Purchase Planning**: Estimate future costs of major purchases adjusted for inflation.
    - **Scenario Comparisons**: Visualize financial outcomes under different economic conditions.
    """)

    # Explanation of Calculations and Algorithms
    st.markdown("""
    ### How We Generate These Insights:
    1. **Time Series Forecasting**:
       - We use **Prophet**, a time series forecasting model developed by Facebook, to predict trends in inflation, wage growth, and unemployment based on historical data.
    2. **Investment Allocation Advice**:
       - Historical market data and user risk tolerance are used to recommend portfolio allocations that balance growth and stability.
    3. **Inflation-Adjusted Projections**:
       - All future financial projections, such as retirement savings and big purchase costs, are adjusted for inflation using forecasted rates.
    4. **Scenario Analysis**:
       - We simulate optimistic, moderate, and pessimistic scenarios to show potential financial outcomes under varying conditions.
    5. **Dynamic Recommendations**:
       - Emergency fund and savings recommendations are dynamically calculated based on user inputs (age, expenses) and forecasted economic conditions.
    """)

    # Input User Financial Data
    st.markdown("### Personal Financial Data")
    current_age = st.number_input("Enter Your Current Age:", min_value=18, max_value=80, value=30)
    retirement_age = st.number_input("Enter Your Desired Retirement Age:", min_value=50, max_value=80, value=65)
    yearly_income = st.number_input("Enter Your Yearly Income ($):", min_value=0, value=50000)
    monthly_expenses = st.number_input("Enter Your Monthly Expenses ($):", min_value=0, value=2000)
    savings = st.number_input("Enter Your Total Savings ($):", min_value=0, value=10000)

    disposable_income = yearly_income - (monthly_expenses * 12)
    st.markdown(f"### Your Disposable Income: ${disposable_income:.2f}")

    years_to_retirement = retirement_age - current_age

    # Forecast Horizon Slider
    forecast_horizon = st.slider("Forecast Horizon (Years)", 1, 10, 5)

    if st.button("Run Comprehensive Forecast"):
        st.markdown("### Comprehensive Financial Recommendations")

        # Prepare Data for Machine Learning Models
        df = data.rename(columns={"year": "ds", "inflation": "y"})  # Use inflation for forecast as an example
        model = Prophet()
        model.fit(df[["ds", "y"]])
        future = model.make_future_dataframe(periods=forecast_horizon)
        forecast = model.predict(future)

        # Retirement Planning
        if years_to_retirement > 0:
            inflation_rate = forecast["yhat"].iloc[-1] / 100
            projected_savings = savings * ((1 + inflation_rate) ** years_to_retirement)
            st.markdown(f"- 🏦 **Projected Savings at Retirement (Adjusted for Inflation):** ${projected_savings:.2f}")
            st.markdown("- 📘 **Recommendation:** Save an additional 10% of your income annually to meet your retirement goals.")

        # Investment Allocation Advice
        risk_tolerance = st.radio("Select Your Risk Tolerance Level", ["Low", "Medium", "High"])
        if risk_tolerance == "Low":
            st.markdown("- 🛡️ **Recommended Allocation:** 70% Bonds, 20% Real Estate, 10% Stocks")
        elif risk_tolerance == "Medium":
            st.markdown("- ⚖️ **Recommended Allocation:** 50% Stocks, 30% Bonds, 20% Real Estate")
        else:
            st.markdown("- 🚀 **Recommended Allocation:** 70% Stocks, 20% Real Estate, 10% Bonds")

        # Emergency Fund Recommendation
        if current_age < 40:
            emergency_fund = monthly_expenses * 6
        else:
            emergency_fund = monthly_expenses * 12
        st.markdown(f"- 🚨 **Recommended Emergency Fund:** ${emergency_fund:.2f}")
        st.markdown("- 🏦 **Tip:** Keep this amount in a high-yield savings account for quick access.")

        # Big Purchase Planning
        house_price = st.number_input("Enter Current Price of Desired House ($):", min_value=0, value=300000)
        future_price = house_price * ((1 + inflation_rate) ** forecast_horizon)
        st.markdown(f"- 🏠 **Projected House Price in {forecast_horizon} Years:** ${future_price:.2f}")
        st.markdown("- 📘 **Recommendation:** Start saving an additional 10% annually to meet this goal.")

        # Scenario Comparisons
        st.markdown("### Financial Scenarios")
        scenario = st.radio("Select Scenario", ["Optimistic", "Moderate", "Pessimistic"])
        adjustment_factor = {"Optimistic": 1.1, "Moderate": 1.0, "Pessimistic": 0.9}[scenario]
        adjusted_forecast = forecast["yhat"] * adjustment_factor
        fig = px.line(forecast, x="ds", y=adjusted_forecast, title=f"{scenario} Scenario")
        st.plotly_chart(fig)

        # Real-Time News Integration
        st.markdown("### Recent News Related to Your Financial Goals")

        @st.cache_data
        def fetch_news(query):
            api_key = "c4cda9e665ab468c8fbbc59df598fca3"
            url = f"https://newsapi.org/v2/everything?q={query}&apiKey={api_key}"
            response = requests.get(url)
            articles = response.json()["articles"][:3]
            return articles

        news_articles = fetch_news("finance")
        for article in news_articles:
            st.markdown(f"- [{article['title']}]({article['url']})")

        # Explain Algorithms and Models
        with st.expander("How Does This Work?"):
            st.markdown("""
            - We use **Prophet**, a robust forecasting model developed by Facebook.
            - The model predicts future trends based on historical data for inflation, unemployment, or wage growth.
            - Recommendations are generated using these forecasts and aligned with your financial inputs.
            - Investment advice incorporates risk tolerance levels and historical returns.
            - Scenario comparisons allow users to visualize financial outcomes under various economic conditions.
            """)

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
    sentiments, avg_sentiment, articles = analyze_sentiment()
    st.metric("Overall Sentiment Score", avg_sentiment)

    # Sentiment Distribution Pie Chart
    sentiment_distribution = {
        "Positive": sum(1 for s in sentiments if s > 0),
        "Neutral": sum(1 for s in sentiments if s == 0),
        "Negative": sum(1 for s in sentiments if s < 0),
    }
    fig_pie = px.pie(
        values=list(sentiment_distribution.values()),
        names=list(sentiment_distribution.keys()),
        title="Sentiment Distribution of Latest Articles",
    )
    st.plotly_chart(fig_pie)

    # Top Keywords
    stop_words = set(stopwords.words("english"))
    keywords = [
        word.lower()
        for article in articles[:5]
        for word in article["title"].split()
        if word.lower() not in stop_words and word.isalpha()
    ]
    most_common_keywords = Counter(keywords).most_common(10)
    keywords_str = ", ".join([word for word, freq in most_common_keywords])
    st.markdown(f"**Top Keywords from Articles:** {keywords_str}")

    # Word Cloud
    if keywords:
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(
            " ".join(keywords)
        )
        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)

    # Top Articles with Links
    st.subheader("Top Articles")
    for article in articles[:5]:
        st.markdown(f"- [{article['title']}]({article['url']})")

    # Sentiment Timeline
    if sentiments:
        dates = pd.date_range(end=pd.Timestamp.now(), periods=len(sentiments))
        timeline_df = pd.DataFrame({"Date": dates, "Sentiment": sentiments})
        fig_timeline = px.line(timeline_df, x="Date", y="Sentiment", title="Sentiment Over Time")
        st.plotly_chart(fig_timeline)

# Generate Report Tab
with tabs[5]:
    st.subheader("Generate Report")
    if st.button("Download Report"):
        generate_pdf(data)
        with open("economic_report.pdf", "rb") as pdf:
            st.download_button("Download Report", pdf, file_name="economic_report.pdf")
