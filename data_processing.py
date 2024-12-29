import sqlite3
import pandas as pd
from sklearn.cluster import KMeans

# Function to fetch economic data
def fetch_data():  # Replace this function in your script
    # Fixed: Ensure all arrays are the same length
    years = list(range(2000, 2023))  # 23 years of data

    # Create dummy data with consistent lengths
    gdp_growth = [-1, 2, 3] * (len(years) // 3)  # Repeat pattern
    gdp_growth += [-1] * (len(years) - len(gdp_growth))  # Pad to match `years`

    inflation = [3, 4, 5] * (len(years) // 3)
    inflation += [3] * (len(years) - len(inflation))  # Pad to match `years`

    unemployment = [5, 6, 7] * (len(years) // 3)
    unemployment += [5] * (len(years) - len(unemployment))  # Pad to match `years`

    # Create the DataFrame
    data = {
        "year": years,
        "GDP YoY Growth (%)": gdp_growth,
        "Inflation Rate (%)": inflation,
        "Unemployment Rate (%)": unemployment,
    }
    return pd.DataFrame(data)

# Function to process and cluster data
def process_data(data):
    data["GDP Rolling Avg (%)"] = data["GDP YoY Growth (%)"].rolling(window=3).mean()
    features = data[["GDP YoY Growth (%)", "Inflation Rate (%)", "Unemployment Rate (%)"]].dropna()
    kmeans = KMeans(n_clusters=3, random_state=42)
    data["Economic Phase"] = kmeans.fit_predict(features)
    data["Economic Phase Name"] = data["Economic Phase"].map({0: "Recession", 1: "Growth", 2: "Stagflation"})
    return data

# Function to write data to SQLite
def write_to_db(data):
    conn = sqlite3.connect("economic_data.db")
    data.to_sql("economic_data", conn, if_exists="replace", index=False)
    conn.close()

# Main script
if __name__ == "__main__":
    try:
        raw_data = fetch_data()  # Fetch the data
        processed_data = process_data(raw_data)  # Process the data
        write_to_db(processed_data)  # Save to SQLite database
        print("Database successfully created!")
    except Exception as e:
        print(f"Error during data processing: {e}")
