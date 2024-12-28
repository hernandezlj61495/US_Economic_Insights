import sqlite3

# Connect to SQLite database
conn = sqlite3.connect('economic_data.db')

# Create table (if it doesn't already exist)
conn.execute('''
CREATE TABLE IF NOT EXISTS economic_data (
    country TEXT,
    year INTEGER,
    "GDP (USD)" REAL,
    "Inflation Rate (%)" REAL,
    "Unemployment Rate (%)" REAL,
    "GDP YoY Growth (%)" REAL
)
''')

conn.close()
print("Database setup complete.")
