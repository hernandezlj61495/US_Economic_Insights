import schedule
import time
import subprocess
import logging

# Setup logging to file and console
logging.basicConfig(
    filename='scheduler.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger().addHandler(console)

# Define the update process
def update_data():
    logging.info("Running data update...")  # Log start of update
    try:
        result = subprocess.run(['python3', 'data_processing.py'], check=True, capture_output=True, text=True)
        logging.info(f"Data updated successfully! Output:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error occurred while updating data: {e.stderr}")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")

# Schedule the update to run every minute for testing
schedule.every().day.at("00:00").do(update_data)  # Change to .day.at("00:00") for daily updates in deployment

logging.info("Scheduler started. Waiting for tasks...")

# Scheduler infinite loop
try:
    while True:
        schedule.run_pending()
        time.sleep(10)  # Adjust sleep interval for quicker testing
except KeyboardInterrupt:
    logging.info("Scheduler stopped manually.")
