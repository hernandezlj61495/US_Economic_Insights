import os
from pathlib import Path
import subprocess
import sqlite3
import pandas as pd
import streamlit as st
import plotly.express as px

# Use Streamlit temp directory for the database
db_path = Path(os.getenv('STREAMLIT_TEMP_DIR', '.')) / "economic_data.db"

if not db_path.exists():
    st.warning("Database file is missing. Generating it now...")
    try:
        result = subprocess.run(
            ["python3", "data_processing.py"],
            check=True,
            capture_output=True,
            text=True
        )
        st.success("Database generated successfully!")
    except subprocess.CalledProcessError as e:
        st.error(f"Failed to generate the database. Error details: {e.stderr}")
        st.stop()

def load_data():
    """Load data from SQLite database."""
    try:
        conn = sqlite3.connect(db_path)
        data = pd.read_sql("SELECT * FROM economic_data", conn)
        conn.close()
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

# Streamlit app
st.title("US Economic Insights Dashboard")

# Load data
data = load_data()

# Economic phases visualization
st.subheader("Clustered Economic Phases")
fig = px.scatter(
    data,
    x="GDP YoY Growth (%)",
    y="Unemployment Rate (%)",
    color="Economic Phase Name",
    title="Clustered Economic Phases"
)
st.plotly_chart(fig)
