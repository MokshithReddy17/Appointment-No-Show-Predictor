from config import *

class InterventionRecommender:
    def __init__(self):
        self.high_risk_threshold = HIGH_RISK_THRESHOLD
        self.medium_risk_threshold = MEDIUM_RISK_THRESHOLD
    
    def recommend_interventions(self, patient_data, risk_score):
        """Recommend interventions based on risk score and patient characteristics"""
        interventions = []
        
        if risk_score >= self.high_risk_threshold:
            interventions.append("Priority: High")
            interventions.append("Send multiple reminders (SMS + call)")
            interventions.append("Schedule transportation if needed")
            interventions.append("Consider flexible rescheduling")
            interventions.append("Personalized message from doctor")
            
            if patient_data.get('prior_no_shows', 0) > 2:
                interventions.append("Flag for case manager review")
                
            if patient_data.get('age', 0) > 65:
                interventions.append("Arrange caregiver communication")
                
        elif risk_score >= self.medium_risk_threshold:
            interventions.append("Priority: Medium")
            interventions.append("Send SMS reminder 2 days before")
            interventions.append("Send email reminder if available")
            
            if patient_data.get('days_until_appointment', 0) > 14:
                interventions.append("Send interim reminder 1 week before")
        else:
            interventions.append("Priority: Low")
            interventions.append("Standard reminder system")
        
        # Weather-specific interventions
        if patient_data.get('bad_weather', 0) == 1:
            interventions.append("Weather alert: Offer rescheduling")
        
        # Time-specific interventions
        if patient_data.get('is_morning', 0) == 1 and risk_score > 0.3:
            interventions.append("Morning appointment: Confirm wake-up call needed")
        
        return interventions