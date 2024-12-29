import os
from pathlib import Path
import sqlite3
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

def fetch_gdp_data():
    """Fetch GDP data (placeholder function)."""
    print("Fetching GDP data...")
    gdp_data = pd.DataFrame({
        "year": range(2000, 2023),
        "GDP YoY Growth (%)": np.random.uniform(-5, 5, 23)
    })
    print("GDP data fetched successfully.")
    return gdp_data

def fetch_inflation_data():
    """Fetch Inflation data (placeholder function)."""
    print("Fetching Inflation data...")
    inflation_data = pd.DataFrame({
        "year": range(2000, 2023),
        "Inflation Rate (%)": np.random.uniform(0, 10, 23)
    })
    print("Inflation data fetched successfully.")
    return inflation_data

def fetch_unemployment_data():
    """Fetch Unemployment data (placeholder function)."""
    print("Fetching Unemployment data...")
    unemployment_data = pd.DataFrame({
        "year": range(2000, 2023),
        "Unemployment Rate (%)": np.random.uniform(3, 15, 23)
    })
    print("Unemployment data fetched successfully.")
    return unemployment_data

def cluster_economic_phases(data):
    """Cluster economic data into phases using KMeans."""
    print("Clustering economic phases...")
    features = data[["GDP YoY Growth (%)", "Inflation Rate (%)", "Unemployment Rate (%)"]]
    kmeans = KMeans(n_clusters=3, random_state=42)
    data["Economic Phase"] = kmeans.fit_predict(features)
    phase_names = {0: "Recession", 1: "Growth", 2: "Stagflation"}
    data["Economic Phase Name"] = data["Economic Phase"].map(phase_names)
    print("Clustering complete.")
    return data

def process_data():
    """Fetch and process economic data."""
    try:
        print("Fetching all data...")
        gdp_data = fetch_gdp_data()
        inflation_data = fetch_inflation_data()
        unemployment_data = fetch_unemployment_data()
        print("All data fetched successfully.")

        print("Merging datasets...")
        data = pd.merge(gdp_data, inflation_data, on="year")
        data = pd.merge(data, unemployment_data, on="year")
        print("Datasets merged successfully.")

        print("Calculating additional metrics...")
        data["GDP Rolling Avg (%)"] = data["GDP YoY Growth (%)"].rolling(3).mean()
        print("Metrics calculated successfully.")

        print("Applying clustering...")
        data = cluster_economic_phases(data)
        print("Clustering applied successfully.")
        
        return data
    except Exception as e:
        print(f"Error during data processing: {e}")
        raise

def save_to_database(data):
    """Save the processed data to SQLite database in a temp directory."""
    try:
        print("Saving data to SQLite database...")
        temp_dir = Path(os.getenv('STREAMLIT_TEMP_DIR', '.'))
        db_path = temp_dir / "economic_data.db"

        conn = sqlite3.connect(db_path)
        data.to_sql("economic_data", conn, if_exists="replace", index=False)
        conn.close()

        print(f"Data saved successfully to {db_path}")
    except Exception as e:
        print(f"Error saving data to database: {e}")
        raise

if __name__ == "__main__":
    try:
        print("Starting data processing...")
        data = process_data()
        print("Data processing complete.")

        save_to_database(data)
        print("Database generation complete.")
    except Exception as e:
        print(f"Critical error in data_processing.py: {e}")
        raise

