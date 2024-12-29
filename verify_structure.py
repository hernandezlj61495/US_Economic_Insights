import os

expected_structure = {
    "files": [
        "README.md",
        "dashboard.py",
        "data_processing.py",
        "db_setup.py",
        "pipeline_scheduler.py",
        "requirements.txt",
        "economic_data.db",
    ],
    "folders": {
        "images": ["correlation_matrix.png", "dashboard_preview.png", "gdp_trends.png"]
    }
}

# Verify files
missing_files = [f for f in expected_structure["files"] if not os.path.isfile(f)]
if missing_files:
    print("Missing files:", missing_files)
else:
    print("All required files are present!")

# Verify folders and their contents
for folder, files in expected_structure["folders"].items():
    if not os.path.isdir(folder):
        print(f"Folder '{folder}' is missing.")
    else:
        for file in files:
            if not os.path.isfile(os.path.join(folder, file)):
                print(f"Missing file: {file} in folder {folder}")

