import schedule
import time
import subprocess

# Define the update process
def update_data():
    subprocess.run(['python3', 'data_processing.py'])
    print("Data updated successfully!")

# Schedule the update to run daily at midnight
schedule.every().day.at("00:00").do(update_data)

print("Scheduler is running. Press Ctrl+C to stop.")
while True:
    schedule.run_pending()
    time.sleep(1)

print("Scheduler is running. Press Ctrl+C to stop.")
print("Waiting for the next scheduled task...")
