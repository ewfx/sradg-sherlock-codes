import pandas as pd
import numpy as np
from datetime import datetime

class DataProcessor:
    """Handles data ingestion, cleaning, preprocessing and feature engineering"""
    
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.required_columns = [
            'AsofDate', 'Company', 'Account', 'AU', 'Currency',
            'Primary Account', 'Secondary Account', 'GL Balance', 'IHUB Balance'
        ]
        self.text_columns = [
            'Company', 'Account', 'AU', 'Currency',
            'Primary Account', 'Secondary Account'
        ]
    
    def load_data(self):
        """Load and validate the Excel file"""
        try:
            self.df = pd.read_excel(self.file_path)
            
            # Validate required columns
            missing_cols = [col for col in self.required_columns if col not in self.df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
                
            return self
        except Exception as e:
            raise ValueError(f"Error loading file: {str(e)}")
    
    def clean_data(self):
        """Perform data cleaning operations"""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        # Convert text columns to string and clean
        for col in self.text_columns:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(str).str.strip()
                
        # Clean numeric columns
        numeric_cols = ['GL Balance', 'IHUB Balance']
        for col in numeric_cols:
            if col in self.df.columns:
                # Remove any non-numeric characters
                self.df[col] = pd.to_numeric(
                    self.df[col].astype(str).str.replace(r'[^\d.-]', '', regex=True),
                    errors='coerce'
                )
                
        # Handle date conversion
        self.df['AsofDate'] = pd.to_datetime(self.df['AsofDate'], errors='coerce')
        
        # Drop rows with critical missing values
        self.df = self.df.dropna(subset=['AsofDate', 'GL Balance', 'IHUB Balance'])
        
        return self
    
    def preprocess_data(self):
        """Perform data preprocessing and feature engineering"""
        if self.df is None:
            raise ValueError("Data not cleaned. Call clean_data() first.")
            
        # Ensure numeric columns have proper precision
        self.df['GL Balance'] = self.df['GL Balance'].round(2)
        self.df['IHUB Balance'] = self.df['IHUB Balance'].round(2)
        
        # Create composite key for grouping
        self.df['CompositeKey'] = (
            self.df['Company'] + '_' + 
            self.df['Account'] + '_' + 
            self.df['AU'] + '_' + 
            self.df['Currency'] + '_' + 
            self.df['Primary Account']
        )
        
        # Sort by composite key and date for proper differencing
        self.df = self.df.sort_values(['CompositeKey', 'AsofDate'])
        
        return self
    
    def calculate_differences(self):
        """Calculate balance differences and match status"""
        if self.df is None:
            raise ValueError("Data not preprocessed. Call preprocess_data() first.")
            
        self.df['Balance Difference'] = self.df['GL Balance'] - self.df['IHUB Balance']
        self.df['Abs Difference'] = self.df['Balance Difference'].abs()
        self.df['Match Status'] = np.where(
            self.df['Abs Difference'] < 1, 'Match', 'Break'
        )
        
        # Add percentage difference
        self.df['Pct Difference'] = np.where(
            self.df['IHUB Balance'] != 0,
            (self.df['Balance Difference'] / self.df['IHUB Balance']) * 100,
            np.nan
        )
        
        return self
    
    def add_previous_differences(self):
        """Add previous balance difference for each group"""
        if self.df is None:
            raise ValueError("Differences not calculated. Call calculate_differences() first.")
            
        # Calculate previous difference using the composite key
        self.df['Previous Balance Difference'] = self.df.groupby('CompositeKey')['Balance Difference'].shift(1)
        self.df['Difference Change'] = self.df['Balance Difference'] - self.df['Previous Balance Difference']
        
        return self
    
    def get_processed_data(self):
        """Return the fully processed dataframe"""
        if self.df is None:
            raise ValueError("Processing not complete. Run all steps first.")
            
        # Drop temporary columns
        result = self.df.drop(columns=['CompositeKey'], errors='ignore')
        
        # Ensure proper column order
        column_order = self.required_columns + [
            'Balance Difference', 'Abs Difference', 'Pct Difference',
            'Previous Balance Difference', 'Difference Change',
            'Match Status'
        ]
        
        return result[[col for col in column_order if col in result.columns]]
    
    def full_pipeline(self):
        """Execute the complete data processing pipeline"""
        return (
            self.load_data()
            .clean_data()
            .preprocess_data()
            .calculate_differences()
            .add_previous_differences()
            .get_processed_data()
        )
