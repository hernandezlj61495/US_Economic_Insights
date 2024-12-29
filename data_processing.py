from pandas_datareader import wb
import pandas as pd
import sqlite3

def fetch_and_process_data():
    try:
        print("Fetching GDP data...")
        gdp_df = wb.download(indicator='NY.GDP.MKTP.CD', country='US', start=2000, end=2022)
        gdp_df.reset_index(inplace=True)
        gdp_df.rename(columns={'NY.GDP.MKTP.CD': 'GDP'}, inplace=True)

        print("Fetching Inflation data...")
        inflation_df = wb.download(indicator='FP.CPI.TOTL', country='US', start=2000, end=2022)
        inflation_df.reset_index(inplace=True)
        inflation_df.rename(columns={'FP.CPI.TOTL': 'Inflation'}, inplace=True)

        print("Fetching Unemployment data...")
        unemployment_df = wb.download(indicator='SL.UEM.TOTL.ZS', country='US', start=2000, end=2022)
        unemployment_df.reset_index(inplace=True)
        unemployment_df.rename(columns={'SL.UEM.TOTL.ZS': 'Unemployment'}, inplace=True)
    except Exception as e:
        print(f"Error occurred while fetching data: {e}")
        return pd.DataFrame()  # Return an empty DataFrame if the fetch fails

    # Merge datasets
    print("Merging datasets...")
    economic_data = pd.merge(gdp_df, inflation_df, on=['country', 'year'], how='outer')
    economic_data = pd.merge(economic_data, unemployment_df, on=['country', 'year'], how='outer')

    # Rename columns for clarity
    print("Renaming columns...")
    economic_data.rename(columns={
        'GDP': 'GDP (USD)',
        'Inflation': 'Inflation Rate (%)',
        'Unemployment': 'Unemployment Rate (%)'
    }, inplace=True)

    # Calculate GDP YoY Growth
    print("Calculating GDP YoY Growth...")
    economic_data['GDP YoY Growth (%)'] = economic_data['GDP (USD)'].pct_change() * 100

    print("Data processing complete.")
    return economic_data

def save_to_database(data):
    try:
        print("Saving data to database...")
        conn = sqlite3.connect('./economic_data.db')
        data.to_sql('economic_data', conn, if_exists='replace', index=False)
        conn.close()
        print("Data saved to database successfully!")
    except Exception as e:
        print(f"Error occurred while saving to database: {e}")

if __name__ == "__main__":
    data = fetch_and_process_data()
    if not data.empty:
        save_to_database(data)
    else:
        print("No data to save. Please check the API or your internet connection.")
