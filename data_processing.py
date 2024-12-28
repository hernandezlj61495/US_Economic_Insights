from pandas_datareader import wb
import pandas as pd
import sqlite3

def fetch_and_process_data():
    # Fetch GDP data
    gdp_df = wb.download(indicator='NY.GDP.MKTP.CD', country='US', start=2000, end=2022)
    gdp_df.reset_index(inplace=True)
    gdp_df.rename(columns={'NY.GDP.MKTP.CD': 'GDP'}, inplace=True)

    # Fetch Inflation data
    inflation_df = wb.download(indicator='FP.CPI.TOTL', country='US', start=2000, end=2022)
    inflation_df.reset_index(inplace=True)
    inflation_df.rename(columns={'FP.CPI.TOTL': 'Inflation'}, inplace=True)

    # Fetch Unemployment data
    unemployment_df = wb.download(indicator='SL.UEM.TOTL.ZS', country='US', start=2000, end=2022)
    unemployment_df.reset_index(inplace=True)
    unemployment_df.rename(columns={'SL.UEM.TOTL.ZS': 'Unemployment'}, inplace=True)

    # Merge datasets
    economic_data = pd.merge(gdp_df, inflation_df, on=['country', 'year'], how='outer')
    economic_data = pd.merge(economic_data, unemployment_df, on=['country', 'year'], how='outer')

    # Rename columns for clarity
    economic_data.rename(columns={
        'GDP': 'GDP (USD)',
        'Inflation': 'Inflation Rate (%)',
        'Unemployment': 'Unemployment Rate (%)'
    }, inplace=True)

    # Calculate GDP YoY Growth
    economic_data['GDP YoY Growth (%)'] = economic_data['GDP (USD)'].pct_change() * 100

    return economic_data

def save_to_database(data):
    # Connect to SQLite database and save data
    conn = sqlite3.connect('economic_data.db')
    data.to_sql('economic_data', conn, if_exists='replace', index=False)
    conn.close()

if __name__ == "__main__":
    # Fetch, process, and save data
    data = fetch_and_process_data()
    save_to_database(data)
    print("Data saved to database!")
