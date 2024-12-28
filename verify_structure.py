import os

# Define the expected structure
expected_structure = {
    "files": [
        "README.md",
        "dashboard.py",
        "data_processing.py",
        "db_setup.py",
        "pipeline_scheduler.py",
        "requirements.txt",
        "visualizations.ipynb",
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
missing_folders = []
missing_folder_files = {}

for folder, files in expected_structure["folders"].items():
    if not os.path.isdir(folder):
        missing_folders.append(folder)
    else:
        missing_files_in_folder = [
            f for f in files if not os.path.isfile(os.path.join(folder, f))
        ]
        if missing_files_in_folder:
            missing_folder_files[folder] = missing_files_in_folder

if missing_folders:
    print("Missing folders:", missing_folders)
if missing_folder_files:
    print("Missing files in folders:", missing_folder_files)
if not missing_folders and not missing_folder_files:
    print("All folders and files are present!")
