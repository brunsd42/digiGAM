import os
import subprocess
from pathlib import Path

def convert_notebooks(list_file_path):
    with open(list_file_path, 'r') as f:
        notebook_paths = [line.strip() for line in f if line.strip()]

    # Define the output directory for Python scripts
    output_dir = Path("../scripts")
    output_dir.mkdir(parents=True, exist_ok=True)

    for notebook_path in notebook_paths:
        notebook = Path(notebook_path)
        if not notebook.exists() or notebook.suffix != '.ipynb':
            print(f"Skipping invalid notebook path: {notebook}")
            continue

        # Convert notebook to Python script in the current directory
        print(f"Converting {notebook} to Python script")
        subprocess.run([
            "jupyter", "nbconvert", "--to", "script", str(notebook)
        ], check=True)

        # Move the generated .py file to ../scripts/
        generated_py = notebook.with_suffix('.py')
        target_py = output_dir / generated_py.name

        if generated_py.exists():
            generated_py.rename(target_py)
            print(f"Moved {generated_py} to {target_py}")
        else:
            print(f"Expected Python script {generated_py} not found after conversion.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python export_notebooks_to_scripts.py <notebook_list.txt>")
    else:
        convert_notebooks(sys.argv[1])