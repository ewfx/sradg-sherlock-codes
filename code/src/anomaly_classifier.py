import numpy as np
import pandas as pd

class HybridAnomalyClassifier:
    """Combines rules and statistical methods for break classification"""
    
    def __init__(self):
        self.thresholds = {
            'significant_variance': 0.5,  # 50% change from previous
            'small_difference': 1.0,      # Absolute difference threshold
            'large_difference': 10000.0   # Large absolute difference
        }
    
    def classify_break(self, row):
        """Classify a single break record"""
        if row['Match Status'] == 'Match':
            return 'Within Tolerance'
        
        current_diff = row['Balance Difference']
        abs_diff = abs(current_diff)
        prev_diff = row.get('Previous Balance Difference', 0)
        abs_prev_diff = abs(prev_diff) if pd.notna(prev_diff) else 0
        
        # Rule-based classification
        if abs_diff < self.thresholds['small_difference']:
            return 'Small Difference'
        
        if pd.isna(prev_diff):
            return 'New Difference'
        
        if abs_diff > self.thresholds['large_difference']:
            return 'Large Difference'
        
        if abs(current_diff - prev_diff) > (self.thresholds['significant_variance'] * abs_prev_diff):
            return 'Significant Variance'
        
        if np.sign(current_diff) != np.sign(prev_diff):
            return 'Direction Change'
        
        if abs(current_diff - prev_diff) < 0.1 * abs_prev_diff:
            return 'Consistent Difference'
        
        return 'Moderate Difference'
    
    def classify_all(self, df):
        """Classify all breaks in the dataframe"""
        if 'Match Status' not in df.columns:
            raise ValueError("Match Status column not found. Run reconciliation first.")
        
        df['Break Classification'] = df.apply(self.classify_break, axis=1)
        return df