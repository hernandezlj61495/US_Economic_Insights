import sqlite3

def setup_database():
    conn = sqlite3.connect('economic_data.db')
    conn.execute('''
    CREATE TABLE IF NOT EXISTS economic_data (
        country TEXT,
        year INTEGER,
        gdp_growth REAL,
        inflation REAL,
        unemployment REAL,
        economic_phase TEXT
    )
    ''')
    conn.close()
    print("Database setup complete.")

if __name__ == "__main__":
    setup_database()
