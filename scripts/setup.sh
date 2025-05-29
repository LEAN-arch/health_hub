#!/bin/bash
echo "Setting up Health Hub virtual environment and installing dependencies..."

# Define virtual environment directory
VENV_DIR="venv_health_hub" # More specific name

# Check if Python3 is available
if ! command -v python3 &> /dev/null
then
    echo "Error: python3 command could not be found. Please install Python 3."
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment in ./${VENV_DIR}..."
python3 -m venv ${VENV_DIR}

# Activate virtual environment
# Source command might differ slightly based on shell (bash/zsh vs fish)
echo "Activating virtual environment..."
source ./${VENV_DIR}/bin/activate

# Check if activation was successful (optional, based on PROMPT_COMMAND or VIRTUAL_ENV var)
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Error: Virtual environment activation failed."
    echo "Please try activating manually: source ./${VENV_DIR}/bin/activate"
    exit 1
fi

echo "Virtual environment activated: $VIRTUAL_ENV"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
REQUIREMENTS_FILE="requirements.txt"
if [ -f "$REQUIREMENTS_FILE" ]; then
    echo "Installing dependencies from ${REQUIREMENTS_FILE}..."
    pip install -r ${REQUIREMENTS_FILE}
    if [ $? -eq 0 ]; then
        echo "Dependencies installed successfully."
    else
        echo "Error: Failed to install dependencies from ${REQUIREMENTS_FILE}."
        echo "Please check the file and try again."
        exit 1
    fi
else
    echo "Warning: ${REQUIREMENTS_FILE} not found. Skipping dependency installation."
    echo "Please ensure you have a requirements.txt file with necessary packages (e.g., streamlit, pandas, geopandas, plotly)."
fi

echo "Setup complete."
echo "To run the application, ensure the virtual environment is active ('source ./${VENV_DIR}/bin/activate') and then use: streamlit run app_home.py"
