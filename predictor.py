import pandas as pd
import numpy as np
from feature_engineer import FeatureEngineer
from config import *
import pickle

class NoShowPredictor:
    def __init__(self):
        self.model = None
        self.feature_engineer = FeatureEngineer()
        
    def load_model(self):
        """Load the trained model"""
        try:
            with open(MODELS_DIR / MODEL_FILE, 'rb') as f:
                self.model = pickle.load(f)
            return True
        except:
            return False
    
    def predict(self, new_data):
        """Predict no-show probability for new data"""
        if not self.load_model():
            raise Exception("Model not found. Train the model first.")
        
        if not self.feature_engineer.load_preprocessor():
            raise Exception("Preprocessor not found. Train the model first.")
        
        # Prepare features
        features = self.feature_engineer.prepare_features(new_data, train=False)
        
        # Make predictions
        probabilities = self.model.predict_proba(features)[:, 1]
        
        return probabilities