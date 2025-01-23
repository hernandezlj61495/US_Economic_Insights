import sqlite3
import pandas as pd
from sklearn.cluster import KMeans
import requests

def fetch_data(live_data=False):
    if live_data:
        try:
            # Example API call to World Bank for live GDP data
            api_url = "https://api.worldbank.org/v2/country/USA/indicator/NY.GDP.MKTP.CD?format=json"
            response = requests.get(api_url)
            response.raise_for_status()
            json_data = response.json()

            # Process the response into a DataFrame
            records = [
                {
                    "year": int(item['date']),
                    "gdp_growth": None,  # Replace with actual growth calculation if available
                    "inflation": None,   # Replace with actual inflation data if available
                    "unemployment": None, # Replace with actual unemployment data if available
                }
                for item in json_data[1] if item.get('value') is not None
            ]
            df = pd.DataFrame(records)
            return df
        except Exception as e:
            print(f"Error fetching live data: {e}")
            return pd.DataFrame()  # Return empty DataFrame on failure
    else:
        # Static dummy data
        years = list(range(2000, 2023))
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
    if df.empty:
        print("No data to process.")
        return df

    df["rolling_avg_gdp"] = df["gdp_growth"].rolling(window=3).mean()
    features = df[["gdp_growth", "inflation", "unemployment"]].dropna()
    kmeans = KMeans(n_clusters=3, random_state=42)
    df["economic_phase"] = kmeans.fit_predict(features)
    df["economic_phase_name"] = df["economic_phase"].map({0: "Recession", 1: "Growth", 2: "Stagflation"})
    return df

def save_to_db(df, data_source="static"):
    if df.empty:
        print("No data to save to the database.")
        return

    df["data_source"] = data_source
    conn = sqlite3.connect("economic_data.db")
    df.to_sql("economic_data", conn, if_exists="replace", index=False)
    conn.close()

def ensure_database(live_data=False):
    db_path = "economic_data.db"
    if not sqlite3.connect(db_path).execute("SELECT name FROM sqlite_master WHERE type='table' AND name='economic_data';").fetchall():
        print("Initializing database...")
        raw_data = fetch_data(live_data)
        processed_data = process_data(raw_data)
        save_to_db(processed_data, data_source="live" if live_data else "static")
    else:
        print("Database already initialized.")

if __name__ == "__main__":
    # Use live data if required; fallback to static otherwise
    use_live_data = input("Fetch live data? (yes/no): ").strip().lower() == "yes"
    ensure_database(live_data=use_live_data)
    print("Database updated successfully!")

if __name__ == "__main__":
    raw_data = fetch_data()
    processed_data = process_data(raw_data)
    save_to_db(processed_data)
    print("Database updated successfully!")
