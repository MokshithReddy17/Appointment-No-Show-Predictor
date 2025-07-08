import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from config import *
import pickle

class ModelTrainer:
    def __init__(self):
        self.model = None
        self.features = [
            'gender', 'age', 'scholarship', 'hypertension',
            'diabetes', 'alcoholism', 'handicap', 'sms_received',
            'days_until_appointment', 'is_weekend'
        ]

    def load_data(self):
        """Load processed data"""
        try:
            return pd.read_pickle(PROCESSED_DATA_DIR / PROCESSED_DATA_FILE)
        except Exception as e:
            print(f"Error loading processed data: {e}")
            return None

    def train_model(self):
        """Train and evaluate the model"""
        df = self.load_data()
        if df is None:
            return False

        # Prepare features and target
        X = pd.get_dummies(df[self.features], columns=['gender'])
        y = df['no_show']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )

        # Train model
        self.model = RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=RANDOM_STATE
        )
        self.model.fit(X_train, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:,1]

        print("\nModel Evaluation:")
        print(classification_report(y_test, y_pred))
        print(f"ROC AUC Score: {roc_auc_score(y_test, y_prob):.2f}")

        # Save model
        self.save_model()
        return True

    def save_model(self):
        """Save the trained model"""
        try:
            with open(MODELS_DIR / MODEL_FILE, 'wb') as f:
                pickle.dump(self.model, f)
            print(f"Model saved to {MODELS_DIR / MODEL_FILE}")
        except Exception as e:
            print(f"Failed to save model: {e}")

    def load_model(self):
        """Load a trained model"""
        try:
            with open(MODELS_DIR / MODEL_FILE, 'rb') as f:
                self.model = pickle.load(f)
            return True
        except:
            return False