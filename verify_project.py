import subprocess
import os

# Step 1: Verify Python and Dependencies
print("Verifying Python installation and dependencies...")
try:
    python_version = subprocess.check_output(['python3', '--version']).decode('utf-8').strip()
    print(f"Python Version: {python_version}")
except Exception as e:
    print(f"Error verifying Python installation: {e}")

# Step 2: Verify Required Packages
required_packages = [
    'streamlit', 'pandas', 'numpy', 'plotly', 'prophet', 'reportlab', 'textblob', 'statsmodels'
]

print("\nVerifying required Python packages...")
for package in required_packages:
    try:
        __import__(package)
        print(f"Package '{package}' is installed.")
    except ImportError:
        print(f"Package '{package}' is NOT installed. Please install using 'pip install {package}'.")

# Step 3: Verify SQLite Database
print("\nVerifying SQLite database...")
if os.path.exists('economic_data.db'):
    print("Database file 'economic_data.db' found.")
else:
    print("Database file 'economic_data.db' NOT found. Please run 'data_processing.py' to generate it.")

# Step 4: Run Streamlit App
print("\nTesting Streamlit app...")
try:
    subprocess.run(['streamlit', 'run', 'dashboard.py'], check=True)
except Exception as e:
    print(f"Error running Streamlit app: {e}")

# Step 5: Test Data Processing Script
print("\nTesting data processing script...")
try:
    subprocess.run(['python3', 'data_processing.py'], check=True)
    print("Data processing script ran successfully.")
except Exception as e:
    print(f"Error running data processing script: {e}")

