import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent.parent

# Data paths
RAW_DATA_DIR = BASE_DIR /'appointment_noshow_predictor'/ 'data' / 'raw'
PROCESSED_DATA_DIR = BASE_DIR /'appointment_noshow_predictor'/ 'data' / 'processed'
MODELS_DIR = BASE_DIR /'appointment_noshow_predictor' / 'models'

# File names
APPOINTMENTS_FILE = 'KaggleV2-May-2016.csv'
PATIENTS_FILE = 'patients.csv'
WEATHER_FILE = 'weather_data.csv'
PROCESSED_DATA_FILE = 'processed_data.pkl'
MODEL_FILE = 'noshow_model.pkl'

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Intervention thresholds
HIGH_RISK_THRESHOLD = 0.7
MEDIUM_RISK_THRESHOLD = 0.4

# Create directories if they don't exist
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)