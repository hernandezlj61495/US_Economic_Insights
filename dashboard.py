import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px

# Load data from SQLite database
def load_data():
    try:
        conn = sqlite3.connect('economic_data.db')
        data = pd.read_sql('SELECT * FROM economic_data', conn)
        conn.close()
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# Load the data
economic_data = load_data()

# Check if data is loaded
if economic_data.empty:
    st.error("No data available. Please ensure the database is populated by running `data_processing.py`.")
else:
    # Dashboard Title
    st.title('US Economic Insights Dashboard')

    # Economic Indicators Section
    st.subheader('Economic Indicators')
    st.write('Visualize trends in GDP, Inflation, and Unemployment.')

    # Interactive GDP Growth Visualization
    fig1 = px.line(economic_data, x='year', y='GDP YoY Growth (%)', title='US GDP Year-over-Year Growth')
    st.plotly_chart(fig1)

    # Interactive Inflation Visualization
    fig2 = px.line(economic_data, x='year', y='Inflation Rate (%)', title='US Inflation Rate')
    st.plotly_chart(fig2)

    # Interactive Unemployment Visualization
    fig3 = px.line(economic_data, x='year', y='Unemployment Rate (%)', title='US Unemployment Rate')
    st.plotly_chart(fig3)

    # Portfolio Growth Simulator
    st.subheader('Portfolio Growth Simulator')
    initial_investment = st.number_input('Initial Investment ($)', min_value=1000, step=100)
    growth_rate = st.slider('Expected Annual Growth Rate (%)', 0, 15, 7)
    years = st.slider('Investment Period (Years)', 1, 50, 10)

    # Calculate Portfolio Growth
    future_value = initial_investment * ((1 + growth_rate / 100) ** years)
    st.write(f'After {years} years, your investment would grow to **${future_value:,.2f}**.')
