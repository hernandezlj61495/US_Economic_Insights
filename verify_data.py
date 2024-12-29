import sqlite3
import pandas as pd

try:
    conn = sqlite3.connect('economic_data.db')
    data = pd.read_sql('SELECT * FROM economic_data', conn)
    if data.empty:
        print("No data found in the database.")
    else:
        print(data.head())
    conn.close()
except Exception as e:
    print(f"Error querying the database: {e}")
