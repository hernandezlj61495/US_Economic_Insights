import schedule
import time
import subprocess

# Define the update process
def update_data():
    print("Running data update...")  # Debugging output
    try:
        subprocess.run(['python3', 'data_processing.py'], check=True)
        print("Data updated successfully!")
    except Exception as e:
        print(f"Error occurred while updating data: {e}")

# Schedule the update to run every minute for testing
schedule.every(1).minutes.do(update_data)

print("Scheduler is running. Press Ctrl+C to stop.")
while True:
    schedule.run_pending()
    print("Waiting for the next scheduled task...")  # Debugging output
    time.sleep(10)  # Adjust the sleep interval for quicker checks
