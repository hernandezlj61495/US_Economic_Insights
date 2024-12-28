import sqlite3
import pandas as pd

# Connect to the database
conn = sqlite3.connect('economic_data.db')

# Query the data
data = pd.read_sql('SELECT * FROM economic_data', conn)

# Display the first few rows
print(data.head())

conn.close()
