"""
Configuration settings for the mortgage rate prediction project.
"""
import os
from datetime import datetime

# Date settings
START_DATE = "1971-04-02"  # Start date for data collection
CURRENT_DATE = datetime.now().strftime("%Y-%m-%d")  # Current date

# FRED API settings
FRED_API_KEY = os.environ.get("FRED_API_KEY", None)  # Get from environment or set here

# File paths
DATA_DIR = "data"
MODELS_DIR = "models"
RESULTS_DIR = "results"
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

# Ensure directories exist
for directory in [DATA_DIR, MODELS_DIR, RESULTS_DIR, PLOTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Dataset paths
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw_mortgage_dataset.csv")
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "full_mortgage_dataset.csv")

# Target variable settings
TARGET_COLUMN = "MORTGAGE30US_diff"  # Use "MORTGAGE30US" for absolute rate prediction

# Model training settings
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 10
SEED = 42

# Feature engineering settings
CREATE_LAG_FEATURES = True
LAG_PERIODS = [1, 2, 4, 8]
CREATE_ROLLING_FEATURES = True
ROLLING_WINDOWS = [3, 8]