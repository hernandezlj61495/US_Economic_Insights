import sqlite3
import pandas as pd
from sklearn.cluster import KMeans

# Replace with actual API fetch
def fetch_data():
    years = list(range(2000, 2023))  # 23 years of data

    # Ensure all arrays match the length of 'years'
    gdp_growth = [-1, 2, 3] * (len(years) // 3) + [-1] * (len(years) % 3)
    inflation = [3, 4, 5] * (len(years) // 3) + [3] * (len(years) % 3)
    unemployment = [5, 6, 7] * (len(years) // 3) + [5] * (len(years) % 3)

    # Create the DataFrame
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
    return df

def save_to_db(df):
    conn = sqlite3.connect("economic_data.db")
    df.to_sql("economic_data", conn, if_exists="replace", index=False)
    conn.close()

if __name__ == "__main__":
    raw_data = fetch_data()
    processed_data = process_data(raw_data)
    save_to_db(processed_data)
    print("Database updated successfully!")
