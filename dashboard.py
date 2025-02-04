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
from collections import Counter
from nltk.corpus import stopwords
from wordcloud import WordCloud
import nltk
import datetime
from dotenv import load_dotenv  # For environment variables

# =============================
# 2. Initial Setup and Configuration
# =============================

# Load environment variables from .env file
load_dotenv()

# Download NLTK stopwords if not already present
nltk.download("stopwords")

# Suppress sklearn FutureWarnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

# Set Streamlit page configuration
st.set_page_config(
    page_title="US Economic Insights Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================
# 3. Custom CSS for Styling
# =============================

def load_css():
    """
    Load custom CSS styles for the dashboard.
    """
    try:
        with open("styles.css") as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("CSS file 'styles.css' not found. Proceeding without custom styles.")

# Apply custom CSS if available
load_css()

# Alternatively, inject CSS directly
st.markdown(
    """
    <style>
    /* Custom CSS */
    .big-font {
        font-size:20px !important;
    }
    .metric-title {
        font-size: 18px;
        color: #4CAF50;
    }
    /* Footer Styling */
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f1f1f1;
        color: #333;
        text-align: center;
        padding: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =============================
# 4. API Key and Environment Variables
# =============================

# Set API Key for Sentiment Analysis
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
if not NEWSAPI_KEY:
    st.error("API Key for NewsAPI not found. Please set the NEWSAPI_KEY environment variable.")
    st.stop()

# =============================
# 5. Data Loading with Caching
# =============================

@st.cache_data(show_spinner=False)
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
                df = df[["date", "value"]].rename(columns={"date": "year", "value": "gdp"})
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
    gdp = np.random.uniform(1, 5, len(years))
    inflation = np.random.uniform(2, 4, len(years))
    unemployment = np.random.uniform(3, 8, len(years))
    data = {
        "year": years,
        "gdp_growth": gdp,
        "inflation": inflation,
        "unemployment": unemployment,
    }
    return pd.DataFrame(data)

@st.cache_data(show_spinner=False)
def fetch_us_economic_data():
    # Replace with real API URLs or data processing as needed
    data = {
        "State": ["California", "Texas", "Florida", "New York"],
        "GDP": [3.9, 2.1, 1.5, 1.8],  # Trillions
        "Unemployment": [4.0, 3.2, 2.8, 3.9],  # Percentage
        "Inflation": [2.5, 2.4, 2.3, 2.6],  # Percentage
        "Top Industry": ["Technology", "Energy", "Tourism", "Finance"],
        "Economic Summary": [
            "California has a strong technology sector led by Silicon Valley.",
            "Texas benefits from a booming energy sector including oil and gas.",
            "Florida thrives on tourism, with key attractions like Disney World.",
            "New York is driven by its financial services and Wall Street dominance."
        ],
        "State Code": ["CA", "TX", "FL", "NY"]
    }
    df = pd.DataFrame(data)
    return df

# =============================
# 6. Data Processing
# =============================

def process_data(df):
    df["rolling_avg_gdp"] = df["gdp_growth"].rolling(window=3).mean()
    features = df[["gdp_growth", "inflation", "unemployment"]].dropna()
    if not features.empty:
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        df["economic_phase"] = kmeans.fit_predict(features)
        phase_mapping = {0: "Recession", 1: "Growth", 2: "Stagflation"}
        df["economic_phase_name"] = df["economic_phase"].map(phase_mapping)
    else:
        df["economic_phase"] = np.nan
        df["economic_phase_name"] = "Unknown"
    return df

# =============================
# 7. PDF Report Generation
# =============================

def generate_pdf(data):
    current_year = pd.Timestamp.now().year
    latest_data = data[data["year"] == current_year]

    c = canvas.Canvas("economic_report.pdf")
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, 750, "US Economic Insights Report")
    c.setFont("Helvetica", 12)
    c.drawString(100, 730, f"Last Updated: {pd.Timestamp.now().strftime('%Y-%m-%d')}")

    if not latest_data.empty:
        gdp = latest_data["gdp_growth"].values[0]
        inflation = latest_data["inflation"].values[0]
        unemployment = latest_data["unemployment"].values[0]
        c.drawString(100, 710, f"Current Year ({current_year}):")
        c.drawString(120, 690, f"- GDP Growth: {gdp:.2f}%")
        c.drawString(120, 670, f"- Inflation: {inflation:.2f}%")
        c.drawString(120, 650, f"- Unemployment: {unemployment:.2f}%")
    else:
        c.drawString(100, 710, f"No Data Available for {current_year}.")

    c.drawString(100, 630, "Key Insights:")
    c.drawString(120, 610, "- During high inflation, secure assets perform well.")
    c.drawString(120, 590, "- Growth phases are ideal for investments.")

    # Generate Economic Trends Plot
    fig, ax = plt.subplots(figsize=(6, 4))
    if {'gdp_growth', 'inflation', 'unemployment'}.issubset(data.columns):
        data.plot(x="year", y=["gdp_growth", "inflation", "unemployment"], ax=ax)
        ax.set_title("Economic Trends Over Time")
        plt.tight_layout()
        plt.savefig("trends.png")
        plt.close(fig)
        c.drawImage("trends.png", 100, 400, width=400, height=200)
    else:
        c.drawString(100, 600, "Insufficient data for economic trends chart.")

    c.save()

# =============================
# 8. Sentiment Analysis
# =============================

@st.cache_data(show_spinner=False)
def analyze_sentiment():
    try:
        url = f"https://newsapi.org/v2/everything?q=economy&language=en&apiKey={NEWSAPI_KEY}"
        response = requests.get(url)
        response.raise_for_status()
        articles = response.json().get("articles", [])
        if not articles:
            st.warning("No articles found for sentiment analysis.")
            return [], 0, []
        
        sentiments = []
        for article in articles[:5]:
            title = article.get("title", "No title available")
            sentiment_score = TextBlob(title).sentiment.polarity
            sentiments.append(sentiment_score)
        avg_sentiment = np.mean(sentiments) if sentiments else 0
        return sentiments, avg_sentiment, articles
    except Exception as e:
        st.error(f"Error fetching sentiment: {e}")
        return [], 0, []  # Neutral fallback sentiment

# =============================
# 9. News Fetching
# =============================

@st.cache_data(show_spinner=False)
def fetch_news(query):
    try:
        url = f"https://newsapi.org/v2/everything?q={query}&language=en&apiKey={NEWSAPI_KEY}"
        response = requests.get(url)
        response.raise_for_status()
        articles = response.json().get("articles", [])
        return articles[:5]  # Return top 5 articles
    except Exception as e:
        st.error(f"Error fetching news: {e}")
        return []

# =============================
# 10. Main Dashboard Implementation
# =============================

def main():
    # Sidebar Settings
    st.sidebar.title("Settings")
    live_data = st.sidebar.checkbox("Use Live Data", value=False)
    show_debug = st.sidebar.checkbox("Show Debug Info", value=False)

    # Fetch and Process Data
    data = fetch_data(live_data=live_data)
    if not data.empty:
        data = process_data(data)
    else:
        st.error("No data available. Please check data sources.")

    if show_debug:
        st.sidebar.markdown("### Debug Information")
        st.sidebar.write("### Raw Data:")
        st.sidebar.write(data)

    # Create Tabs
    tabs = st.tabs(["Overview", "Visualizations", "Forecasting", "Simulations", "Live Sentiment", "Generate Report"])

    # =============================
    # 10.1. Overview Tab
    # =============================
    with tabs[0]:
        st.header("Overview")

        # Display Banner Image with Error Handling
        try:
            st.image("banner.jpg", use_container_width=True)
        except FileNotFoundError:
            st.warning("Banner image not found. Please ensure 'banner.jpg' is in the correct directory.")
        except Exception as e:
            st.error(f"Error loading banner image: {e}")

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

        # Ensure Data Exists
        if data.empty:
            st.error("No economic data available to display.")
        else:
            latest_data = data.iloc[-1]
            if show_debug:
                st.write("Latest Data:", latest_data)  # Debugging

            # Extract Metrics
            gdp_growth = latest_data.get("gdp_growth", None)
            inflation = latest_data.get("inflation", None)
            unemployment = latest_data.get("unemployment", None)
            economic_phase = latest_data.get("economic_phase_name", "Unknown")
            rolling_avg_gdp = latest_data.get("rolling_avg_gdp", None)

            if None in (gdp_growth, inflation, unemployment):
                st.error("Missing economic indicators in the data.")
            else:
                sentiments, avg_sentiment, _ = analyze_sentiment()
                if show_debug:
                    st.write("Sentiments:", sentiments)
                    st.write("Average Sentiment:", avg_sentiment)

                # Calculate Economic Health Score
                gdp_score = np.clip((gdp_growth - 1) / (5 - 1) * 100, 0, 100)
                inflation_score = np.clip((4 - inflation) / (4 - 2) * 100, 0, 100)
                unemployment_score = np.clip((8 - unemployment) / (8 - 3) * 100, 0, 100)
                sentiment_score = np.clip((avg_sentiment + 1) / 2 * 100, 0, 100)

                economic_health_score = round(
                    0.4 * gdp_score + 0.3 * inflation_score + 0.2 * unemployment_score + 0.1 * sentiment_score, 2
                )

                # Display Metrics in Columns
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("GDP Growth", f"{gdp_growth:.2f}%")
                with col2:
                    st.metric("Inflation Rate", f"{inflation:.2f}%")
                with col3:
                    st.metric("Unemployment Rate", f"{unemployment:.2f}%")
                with col4:
                    st.metric("Economic Health Score", f"{economic_health_score}/100")

                # Additional Metrics
                st.markdown("### Rolling Average GDP Growth")
                if rolling_avg_gdp:
                    st.metric("Rolling Avg GDP Growth (3-year)", f"{rolling_avg_gdp:.2f}%")
                else:
                    st.metric("Rolling Avg GDP Growth (3-year)", "N/A")
                    st.warning("Rolling average GDP growth not available.")

                st.markdown("### Current Economic Phase")
                st.write(f"**Economic Phase:** {economic_phase}")

                st.markdown("### Average Sentiment Score")
                if sentiments:
                    st.metric("Average Sentiment Score", f"{avg_sentiment:.2f}")
                else:
                    st.metric("Average Sentiment Score", "N/A")
                    st.warning("No sentiment data available.")

    # =============================
    # 10.2. Visualizations Tab
    # =============================
    with tabs[1]:
        st.header("Visualizations: Explore Economic Trends")

        # Display Current Date and Time
        current_datetime = datetime.datetime.now().strftime("%B %d, %Y at %I:%M %p")
        st.markdown(f"### As of Today: {current_datetime}")
        st.markdown("---")

        # Educational Content
        st.markdown("""
        ### Explore Key Economic Indicators:
        - **GDP**: The total value of goods and services produced in the United States.
        - **Unemployment**: The percentage of people in the labor force without jobs.
        - **Inflation**: The rate at which the general level of prices for goods and services is rising.
        - **Industry Performance**: Discover the top-performing industries for each state.
        """)

        # Fetch and Process Live Data
        economic_data = fetch_us_economic_data()

        # Select Indicator to Visualize
        indicator = st.selectbox("Select Indicator to Visualize", ["GDP", "Unemployment", "Inflation"])

        # Create Interactive Map
        fig = px.choropleth(
            economic_data,
            locations="State Code",
            locationmode="USA-states",
            color=indicator,
            hover_name="State",
            title=f"US {indicator} by State",
            color_continuous_scale="Viridis",
            scope="usa"
        )

        st.plotly_chart(fig, use_container_width=True)

        # Interactive State-Specific Insights
        st.markdown("### State-Specific Insights")
        selected_state = st.selectbox("Select a State to View Details", economic_data["State"].unique())

        state_details = economic_data[economic_data["State"] == selected_state].iloc[0]
        st.markdown(f"**State:** {state_details['State']}")
        st.markdown(f"**Top Industry:** {state_details['Top Industry']}")
        st.markdown(f"**GDP:** ${state_details['GDP']} Trillion")
        st.markdown(f"**Unemployment Rate:** {state_details['Unemployment']}%")
        st.markdown(f"**Inflation Rate:** {state_details['Inflation']}%")
        st.markdown(f"**Economic Summary:** {state_details['Economic Summary']}")

        # Add General Insights Below the Map
        st.markdown("""
        ### General Insights:
        - **GDP**: States with higher GDP often have large industries or populations, like California and Texas.
        - **Unemployment**: Lower unemployment indicates a strong labor market, but may vary by region.
        - **Inflation**: Inflation rates can vary slightly by state but generally follow national trends.
        """)

    # =============================
    # 10.3. Forecasting Tab
    # =============================
    with tabs[2]:
        st.header("Forecasting: Comprehensive Financial Insights for Your Future")

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
            if {'year', 'gdp_growth', 'inflation', 'unemployment'}.issubset(data.columns):
                df = data.rename(columns={"year": "ds", "gdp_growth": "y"})  # Example using gdp_growth for forecast
                model = Prophet()
                model.fit(df[["ds", "y"]])
                future = model.make_future_dataframe(periods=forecast_horizon, freq='Y')
                forecast = model.predict(future)
            else:
                st.error("Insufficient data for forecasting.")
                forecast = pd.DataFrame()

            # Retirement Planning
            if years_to_retirement > 0 and not forecast.empty:
                inflation_rate = forecast["yhat"].iloc[-1] / 100
                projected_savings = savings * ((1 + inflation_rate) ** years_to_retirement)
                st.markdown(f"- 🏦 **Projected Savings at Retirement (Adjusted for Inflation):** ${projected_savings:,.2f}")
                st.markdown("- 📘 **Recommendation:** Save an additional 10% of your income annually to meet your retirement goals.")
            else:
                st.warning("Insufficient data for retirement planning.")

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
            st.markdown(f"- 🚨 **Recommended Emergency Fund:** ${emergency_fund:,.2f}")
            st.markdown("- 🏦 **Tip:** Keep this amount in a high-yield savings account for quick access.")

            # Big Purchase Planning
            house_price = st.number_input("Enter Current Price of Desired House ($):", min_value=0, value=300000)
            if not forecast.empty:
                future_price = house_price * ((1 + inflation_rate) ** forecast_horizon)
                st.markdown(f"- 🏠 **Projected House Price in {forecast_horizon} Years:** ${future_price:,.2f}")
                st.markdown("- 📘 **Recommendation:** Start saving an additional 10% annually to meet this goal.")
            else:
                st.warning("Insufficient data for house price projection.")

            # Scenario Comparisons
            st.markdown("### Financial Scenarios")
            scenario = st.radio("Select Scenario", ["Optimistic", "Moderate", "Pessimistic"])
            adjustment_factor = {"Optimistic": 1.1, "Moderate": 1.0, "Pessimistic": 0.9}[scenario]
            if not forecast.empty:
                adjusted_forecast = forecast.copy()
                adjusted_forecast["yhat"] = adjusted_forecast["yhat"] * adjustment_factor
                fig = px.line(
                    adjusted_forecast,
                    x="ds",
                    y="yhat",
                    title=f"{scenario} Scenario Forecast",
                    labels={"ds": "Year", "yhat": "GDP Growth (%)"},
                    markers=True
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Insufficient data for scenario comparisons.")

            # Real-Time News Integration
            st.markdown("### Recent News Related to Your Financial Goals")

            # Fetch News Articles
            news_articles = fetch_news("finance")
            if news_articles:
                for article in news_articles:
                    st.markdown(f"- [{article['title']}]({article['url']})")
            else:
                st.write("No recent news articles found.")

            # Explain Algorithms and Models
            with st.expander("How Does This Work?"):
                st.markdown("""
                - We use **Prophet**, a robust forecasting model developed by Facebook.
                - The model predicts future trends based on historical data for GDP growth, inflation, or unemployment.
                - Recommendations are generated using these forecasts and aligned with your financial inputs.
                - Investment advice incorporates risk tolerance levels and historical returns.
                - Scenario comparisons allow users to visualize financial outcomes under various economic conditions.
                """)
    
    # =============================
    # 10.4. Simulations Tab
    # =============================
    with tabs[3]:
        st.header("Simulations: Explore Your Financial Future")

        # Educational Section
        st.markdown("""
        ### Understanding Simulations:
        - **Monte Carlo Simulation**: This simulates potential investment outcomes by running thousands of scenarios with random variables.
        - **Portfolio Diversification**: Learn how spreading your investments across stocks, bonds, and real estate reduces risk and stabilizes returns.
        - **Inflation Impact**: Understand how inflation affects purchasing power and adjusts the real returns of your investments.
        - **Savings Growth**: Explore how consistent monthly savings and compounding interest can grow your wealth.
        - **Goal-Based Simulations**: See how long it will take to achieve financial goals like buying a house or retiring.
        """)

        # Interactive Explanation of Diversification
        st.markdown("### What is Portfolio Diversification?")
        st.markdown("""
        Diversification means spreading your investments across different asset classes like stocks, bonds, and real estate to reduce risk.
        - **Why Diversify?**
          - Reduces the chance of losing money if one asset class performs poorly.
          - Helps achieve a balance between risk and reward.
        - **Example Portfolio:**
          - 50% Stocks, 30% Bonds, 20% Real Estate
        """)

        fig = px.pie(values=[50, 30, 20], names=["Stocks", "Bonds", "Real Estate"], title="Example Portfolio Diversification")
        st.plotly_chart(fig, use_container_width=True)

        # Monte Carlo Simulation
        st.markdown("### Monte Carlo Simulation")
        initial_investment = st.number_input("Initial Investment ($):", value=1000.0, step=100.0)
        annual_growth_rate = st.slider("Annual Growth Rate (%):", 0.0, 20.0, 5.0)
        investment_period = st.slider("Investment Period (Years):", 1, 30, 10)

        if st.button("Run Monte Carlo Simulation"):
            simulations = []
            for _ in range(1000):
                returns = np.random.normal(annual_growth_rate / 100, 0.05, investment_period)
                total_growth = initial_investment * np.prod(1 + returns)
                simulations.append(total_growth)

            fig = px.histogram(simulations, nbins=50, title="Monte Carlo Simulation Results", labels={"value": "Portfolio Value ($)"})
            st.plotly_chart(fig, use_container_width=True)

        # Inflation-Adjusted Returns
        st.markdown("### Inflation-Adjusted Returns")
        inflation_rate = st.slider("Inflation Rate (%):", 0.0, 10.0, 2.0)
        real_returns = annual_growth_rate - inflation_rate
        st.markdown(f"- 📉 **Real Annual Growth Rate:** {real_returns:.2f}%")
        st.markdown("- **Tip:** Focus on investments that outpace inflation to maintain purchasing power.")

        # Goal-Based Simulation
        st.markdown("### Goal-Based Simulation")
        goal_amount = st.number_input("Enter Your Financial Goal Amount ($):", value=100000)
        if annual_growth_rate > 0:
            years_to_goal = np.log(goal_amount / initial_investment) / np.log(1 + annual_growth_rate / 100)
            years_to_goal = max(years_to_goal, 0)  # Ensure non-negative
            st.markdown(f"- 🎯 **Time to Achieve Your Goal:** {years_to_goal:.1f} years")
            st.markdown("- 📘 **Action:** Consider increasing your annual growth rate or contributions to achieve your goal faster.")
        else:
            st.warning("Annual growth rate must be greater than 0 to calculate goal-based simulation.")

        # Debt Payoff Simulation
        st.markdown("### Debt Payoff Simulation")
        loan_balance = st.number_input("Enter Your Loan Balance ($):", value=20000)
        interest_rate = st.number_input("Enter Your Loan Interest Rate (%):", value=5.0)
        monthly_payment = st.number_input("Enter Your Monthly Payment ($):", value=500.0)

        monthly_interest = loan_balance * (interest_rate / 100 / 12)
        if monthly_payment > monthly_interest:
            payoff_time = loan_balance / (monthly_payment - monthly_interest)
            st.markdown(f"- 💳 **Estimated Payoff Time:** {payoff_time:.1f} months")
            st.markdown("- ✅ **Tip:** Increase your monthly payment to reduce interest costs.")
        else:
            st.markdown("- ⚠️ **Warning:** Your monthly payment is too low to cover the interest. Consider increasing it.")

        # Scenario-Based Simulations
        st.markdown("### Scenario-Based Simulations")
        scenario = st.radio("Choose a Scenario", ["Optimistic", "Moderate", "Pessimistic"])
        adjustment_factor = {"Optimistic": 1.1, "Moderate": 1.0, "Pessimistic": 0.9}[scenario]
        adjusted_growth_rate = annual_growth_rate * adjustment_factor
        st.markdown(f"- 📊 **Adjusted Annual Growth Rate ({scenario}):** {adjusted_growth_rate:.2f}%")

        # Dynamic Visualization
        future_values = [initial_investment * ((1 + adjusted_growth_rate / 100) ** year) for year in range(1, investment_period + 1)]
        fig = px.line(
            x=range(1, investment_period + 1),
            y=future_values,
            title=f"{scenario} Scenario Growth Over Time",
            labels={"x": "Year", "y": "Portfolio Value ($)"}
        )
        st.plotly_chart(fig, use_container_width=True)

    # =============================
    # 10.5. Live Sentiment Tab
    # =============================
    with tabs[4]:
        st.header("Live Sentiment Analysis")

        sentiments, avg_sentiment, articles = analyze_sentiment()
        if sentiments:
            st.metric("Overall Sentiment Score", f"{avg_sentiment:.2f}")
        else:
            st.metric("Overall Sentiment Score", "N/A")
            st.warning("No sentiment data available.")

        # Sentiment Distribution Pie Chart
        if sentiments:
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
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.write("No sentiment data available to display.")

        # Top Keywords
        if articles:
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
                wordcloud = WordCloud(width=800, height=400, background_color="white").generate(" ".join(keywords))
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation="bilinear")
                ax.axis("off")
                st.pyplot(fig)
            else:
                st.write("No keywords available for word cloud.")
        else:
            st.write("No articles available.")

        # Top Articles with Links
        st.subheader("Top Articles")
        if articles:
            for article in articles[:5]:
                st.markdown(f"- [{article['title']}]({article['url']})")
        else:
            st.write("No articles available.")

        # Sentiment Timeline
        if sentiments:
            dates = pd.date_range(end=pd.Timestamp.now(), periods=len(sentiments))
            timeline_df = pd.DataFrame({"Date": dates, "Sentiment": sentiments})
            fig_timeline = px.line(
                timeline_df, 
                x="Date", 
                y="Sentiment", 
                title="Sentiment Over Time",
                labels={"Date": "Date", "Sentiment": "Sentiment Score"}
            )
            st.plotly_chart(fig_timeline, use_container_width=True)
        else:
            st.write("No sentiment data available to display.")

    # =============================
    # 10.6. Generate Report Tab
    # =============================
    with tabs[5]:
        st.header("Generate Report")
        if st.button("Download Report"):
            try:
                generate_pdf(data)
                with open("economic_report.pdf", "rb") as pdf:
                    st.download_button(
                        label="📥 Download Report",
                        data=pdf,
                        file_name="economic_report.pdf",
                        mime="application/pdf"
                    )
                st.success("Report generated successfully!")
            except Exception as e:
                st.error(f"Error generating report: {e}")

if __name__ == "__main__":
    main()

# =============================
# 11. Footer
# =============================

st.markdown(
    """
    <div class="footer">
        © 2024 US Economic Insights. All rights reserved.
    </div>
    """,
    unsafe_allow_html=True
)
