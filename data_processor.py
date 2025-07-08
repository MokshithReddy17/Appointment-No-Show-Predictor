import pandas as pd
from datetime import datetime
from config import *

class DataProcessor:
    def __init__(self):
        self.df = None

    def load_data(self):
        """Load and preprocess the Kaggle dataset"""
        try:
            # Load the Kaggle dataset with explicit datetime parsing
            self.df = pd.read_csv(
                RAW_DATA_DIR / APPOINTMENTS_FILE,
                parse_dates=['ScheduledDay', 'AppointmentDay'],
                infer_datetime_format=True,
                encoding='utf-8'
            )
            
            # Verify datetime conversion
            if not (pd.api.types.is_datetime64_any_dtype(self.df['ScheduledDay']) and 
                pd.api.types.is_datetime64_any_dtype(self.df['AppointmentDay'])):
                raise ValueError("Date columns not properly converted to datetime")
            
            # Clean column names
            self.df.columns = self.df.columns.str.lower()
            self.df.rename(columns={
                'no-show': 'no_show',
                'hipertension': 'hypertension',
                'handcap': 'handicap'
            }, inplace=True)
            
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False

    def clean_data(self):
        """Clean and transform the data"""
        try:
            # Convert no_show to binary
            self.df['no_show'] = self.df['no_show'].map({'Yes': 1, 'No': 0})
            
            # Ensure datetime format
            self.df['scheduledday'] = pd.to_datetime(self.df['scheduledday'])
            self.df['appointmentday'] = pd.to_datetime(self.df['appointmentday'])
            
            # Calculate days between scheduling and appointment
            self.df['days_until_appointment'] = (
                (self.df['appointmentday'] - self.df['scheduledday']).dt.days
            )
            
            # Remove invalid records
            self.df = self.df[self.df['days_until_appointment'] >= 0]
            self.df = self.df[self.df['age'] >= 0]
            
            # Create time-based features
            self.df['appointment_dow'] = self.df['appointmentday'].dt.day_name()
            self.df['is_weekend'] = self.df['appointmentday'].dt.dayofweek.isin([5,6]).astype(int)
            
            return True
        except Exception as e:
            print(f"Error cleaning data: {e}")
            return False

    def process(self):
        """Complete data processing pipeline"""
        if not self.load_data():
            print("Failed to load data")
            return None
        
        if not self.clean_data():
            print("Failed to clean data")
            return None
            
        # Save processed data
        try:
            self.df.to_pickle(PROCESSED_DATA_DIR / PROCESSED_DATA_FILE)
            return self.df
        except Exception as e:
            print(f"Failed to save processed data: {e}")
            return None