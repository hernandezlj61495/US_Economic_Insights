# US Economic Insights Dashboard

## **Overview**
The **US Economic Insights Dashboard** is a comprehensive tool designed to analyze and visualize key economic indicators. It integrates advanced analytics, machine learning, and data visualization to deliver actionable insights and simulate financial scenarios.

This project showcases expertise in data science, machine learning, and software development by combining time-series forecasting, Monte Carlo simulations, real-time sentiment analysis, and dynamic PDF reporting into a seamless user experience.

---

## **Features**

### **1. Dynamic Visualizations**
- Interactive charts for GDP growth, inflation, and unemployment trends.
- Clustered economic phases to identify distinct historical patterns (e.g., recession, recovery, growth).

### **2. Advanced Forecasting**
- **Prophet** and **ARIMA** models for predicting GDP growth, inflation, and unemployment trends.
- Includes confidence intervals and detailed predictive analytics.

### **3. Portfolio Growth Simulation**
- Monte Carlo simulations for multi-asset portfolios.
- Features include:
  - Inflation-adjusted returns.
  - Market condition toggles (Bullish, Neutral, Bearish).

### **4. Live News Sentiment Analysis**
- Real-time economic news fetched via **NewsAPI**.
- Sentiment analysis using **TextBlob** (Positive, Neutral, Negative).

### **5. Professional PDF Reports**
- Dynamically generated reports summarizing:
  - Key economic metrics.
  - Forecasting insights.
  - Portfolio simulation results.
- Styled for professional use.

### **6. Interactive Glossary**
- Definitions of key economic terms (e.g., GDP, inflation, recession).
- Accessible via collapsible sections.

---

## **Technologies Used**
- **Programming**: Python
- **Libraries**: 
  - **Data Processing**: `pandas`, `numpy`, `scikit-learn`
  - **Visualization**: `plotly`, `streamlit`
  - **Forecasting**: `prophet`, `statsmodels` (ARIMA)
  - **NLP**: `textblob`
  - **PDF Generation**: `reportlab`
  - **API Integration**: `requests`
- **Database**: SQLite
- **Deployment**: Streamlit Cloud
- **Version Control**: GitHub

---

## **Setup**

### **Prerequisites**
1. Python 3.8 or later installed on your system.
2. Install required libraries:
   ```bash
   pip install -r requirements.txt
