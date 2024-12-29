import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import os

# Load data from SQLite database
def load_data():
    try:
        # Debugging: Print current working directory
        st.write(f"Current working directory: {os.getcwd()}")

        # Check if the database exists
        if not os.path.exists('economic_data.db'):
            st.error("Database file 'economic_data.db' is missing. Please run `data_processing.py` to generate it.")
            st.stop()

        # Connect to the database
        conn = sqlite3.connect('./economic_data.db')
        data = pd.read_sql('SELECT * FROM economic_data', conn)
        conn.close()
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# Load the data
economic_data = load_data()

# Debugging: Preview the data
st.write("Loaded data preview:")
st.write(economic_data.head())

# Check if the data is empty
if economic_data.empty:
    st.error("No data available. Please populate the database by running `data_processing.py`.")
    st.stop()
else:
    # Main Dashboard
    st.title('US Economic Insights Dashboard')
    st.subheader('Economic Indicators')
    st.write('Visualize trends in GDP, Inflation, and Unemployment.')

    # GDP Year-over-Year Growth
    fig1 = px.line(economic_data, x='year', y='GDP YoY Growth (%)', title='US GDP Year-over-Year Growth')
    st.plotly_chart(fig1)

    # Inflation Rate
    fig2 = px.line(economic_data, x='year', y='Inflation Rate (%)', title='US Inflation Rate')
    st.plotly_chart(fig2)

    # Unemployment Rate
    fig3 = px.line(economic_data, x='year', y='Unemployment Rate (%)', title='US Unemployment Rate')
    st.plotly_chart(fig3)

    # Portfolio Growth Simulator
    st.subheader('Portfolio Growth Simulator')
    initial_investment = st.number_input('Initial Investment ($)', min_value=1000, step=100)
    growth_rate = st.slider('Expected Annual Growth Rate (%)', 0, 15, 7)
    years = st.slider('Investment Period (Years)', 1, 50, 10)

    # Future Value Calculation
    future_value = initial_investment * ((1 + growth_rate / 100) ** years)
    st.write(f'After {years} years, your investment would grow to **${future_value:,.2f}**.')
