import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from config import *
import pickle

class FeatureEngineer:
    def __init__(self):
        self.preprocessor = None
        self.categorical_features = ['gender', 'neighborhood', 'appointment_type', 'day_of_week']
        self.numerical_features = ['age', 'days_until_appointment', 'prior_no_shows', 
                                 'temperature', 'precipitation', 'humidity']
        
    def create_features(self, df):
        """Create additional features"""
        df['day_of_week'] = df['appointment_date'].dt.day_name()
        df['prior_no_shows'] = df.groupby('patient_id')['no_show'].transform(lambda x: x.shift().expanding().sum())
        df['prior_no_shows'] = df['prior_no_shows'].fillna(0)
        df['is_morning'] = (df['appointment_time'].str.split(':').str[0].astype(int) < 12).astype(int)
        df['is_weekend'] = df['appointment_date'].dt.dayofweek.isin([5, 6]).astype(int)
        df['precipitation'] = df['precipitation'].fillna(0)
        df['bad_weather'] = ((df['precipitation'] > 0.5) | (df['temperature'] < 32) | (df['temperature'] > 90)).astype(int)
        return df
    
    def build_preprocessor(self, df):
        """Build and FIT the preprocessor"""
        numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
        categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numerical_features),
                ('cat', categorical_transformer, self.categorical_features)
            ])
        
        # FIT the preprocessor
        self.preprocessor.fit(df)
        return self.preprocessor
    
    def prepare_features(self, df, train=True):
        """Full feature engineering pipeline"""
        df = self.create_features(df)
        
        if train:
            self.build_preprocessor(df)
            self.save_preprocessor()
        
        # Only transform after fitting
        features = self.preprocessor.transform(df)
        return features
    
    def save_preprocessor(self):
        with open(MODELS_DIR / 'preprocessor.pkl', 'wb') as f:
            pickle.dump(self.preprocessor, f)
    
    def load_preprocessor(self):
        try:
            with open(MODELS_DIR / 'preprocessor.pkl', 'rb') as f:
                self.preprocessor = pickle.load(f)
            return True
        except:
            return False