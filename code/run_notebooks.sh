#!/bin/zsh


# Check if the correct number of arguments is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 /path/to/venv/bin/activate"
    exit 1
fi

# Path to the virtual environment activation script
venv_activate_path="$1"


# File containing the notebook names
notebooks_file="notebooks.txt"

# Activate the environment
source "$venv_activate_path"

# Uninstall existing packages
pip uninstall -y notebook jupyter_contrib_nbextensions

# Reinstall the required modules with specific versions
pip install notebook==6.4.12 jupyter_contrib_nbextensions==0.5.1


# Read the notebook names from the file
notebooks=()
while IFS= read -r line
do
    notebooks+=("$line")
done < "$notebooks_file"

# Iterate over the notebooks
for notebook in "${notebooks[@]}"
do
    notebook=$(echo "$notebook" | tr -d '\n')  # Remove any newline characters
    echo "Running $notebook..."
    jupyter nbconvert --execute --inplace "$notebook"
    exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo "Error encountered in $notebook. Exiting..."
        exit $exit_code
    fi
done

echo "All notebooks executed successfully."
deactivate