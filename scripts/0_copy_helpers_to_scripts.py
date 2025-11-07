import shutil
from pathlib import Path

# Define source and destination directories
source_dir = Path(".")  # current folder (notebooks folder)
dest_dir = Path("../scripts")
dest_dir.mkdir(parents=True, exist_ok=True)

# List of files to copy
files_to_copy = ["config_GAM2025.py", "security_config.py", "functions.py", "test_functions.py"]

# Copy each file
for filename in files_to_copy:
    src_file = source_dir / filename
    dest_file = dest_dir / filename
    if src_file.exists():
        shutil.copy2(src_file, dest_file)
        print(f"Copied {src_file} to {dest_file}")
    else:
        print(f"File {src_file} not found, skipping.")