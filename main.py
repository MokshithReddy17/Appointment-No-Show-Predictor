from data_processor import DataProcessor
from model_trainer import ModelTrainer
import pandas as pd
from config import *

def main():
    print("Medical Appointment No-Show Prediction System")
    print("=" * 60)
    
    # Step 1: Data Processing
    print("\n[1/3] Processing Kaggle dataset...")
    processor = DataProcessor()
    data = processor.process()
    
    if data is None:
        print("\nFailed to process data. Please check:")
        print(f"- Dataset exists at: {RAW_DATA_DIR / APPOINTMENTS_FILE}")
        print("- File permissions are correct")
        return
    
    print(f"\nSuccessfully processed {len(data)} records")
    print("No-show distribution:")
    print(data['no_show'].value_counts())
    
    # Step 2: Model Training
    print("\n[2/3] Training prediction model...")
    trainer = ModelTrainer()
    if not trainer.train_model():
        print("\nModel training failed")
        return
    
    # Step 3: Make Sample Predictions
    print("\n[3/3] Making sample predictions...")
    sample = data.sample(5)
    X_pred = pd.get_dummies(sample[trainer.features], columns=['gender'])
    
    probabilities = trainer.model.predict_proba(X_pred)[:,1]
    
    print("\nSample Predictions:")
    for i, (_, row) in enumerate(sample.iterrows()):
        print(f"\nPatient ID: {row['patientid']}")
        print(f"Gender: {row['gender']}, Age: {row['age']}")
        print(f"Appointment Day: {row['appointmentday'].date()}")
        print(f"Days Until Appointment: {row['days_until_appointment']}")
        print(f"No-Show Probability: {probabilities[i]:.1%}")
        
        if probabilities[i] > 0.7:
            print("RECOMMENDATION: High risk - call patient directly")
        elif probabilities[i] > 0.4:
            print("RECOMMENDATION: Medium risk - send SMS reminder")
        else:
            print("RECOMMENDATION: Low risk - standard email reminder")

if __name__ == "__main__":
    main()