from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

class AnomalyDetector:
    """Statistical anomaly detection component"""
    
    def __init__(self, contamination=0.05):
        self.model = IsolationForest(
            contamination=contamination, 
            random_state=42,
            n_estimators=100
        )
        self.scaler = StandardScaler()
    
    def detect_anomalies(self, df):
        """Detect statistical anomalies in the breaks"""
        breaks_df = df[df['Match Status'] == 'Break'].copy()
        
        if breaks_df.empty:
            df['Is Anomaly'] = 0
            return df
        
        # Prepare features for anomaly detection
        features = breaks_df[['Balance Difference', 'Previous Balance Difference']].fillna(0)
        scaled_features = self.scaler.fit_transform(features)
        
        # Predict anomalies
        breaks_df['Is Anomaly'] = self.model.fit_predict(scaled_features)
        breaks_df['Is Anomaly'] = breaks_df['Is Anomaly'].map({1: 0, -1: 1})  # Convert to binary
        
        # Update original dataframe
        df = df.merge(
            breaks_df[['Is Anomaly']],
            left_index=True,
            right_index=True,
            how='left',
            suffixes=('', '_y')
        )
        df['Is Anomaly'] = df['Is Anomaly'].fillna(0).astype(int)
        
        return df
    
    def calculate_anomaly_scores(self, df):
        """Calculate anomaly scores for all records"""
        features = df[['Balance Difference', 'Previous Balance Difference']].fillna(0)
        if len(features) == 0:
            df['Anomaly Score'] = 0
            return df
            
        scaled_features = self.scaler.transform(features)
        df['Anomaly Score'] = -self.model.score_samples(scaled_features)  # Higher score = more anomalous
        return df