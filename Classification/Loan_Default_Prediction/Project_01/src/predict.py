# FILE: src/predict.py

import pandas as pd             
import joblib                   
import numpy as np              
import re                       
from typing import Dict, Any    

class PredictionHandler:
    
    def __init__(self, model_path: str, mean_map_path: str, encoder_path: str):
        
        try:
            self.model = joblib.load(model_path)
            self.imputation_maps = joblib.load(mean_map_path) 
            self.target_encoder = joblib.load(encoder_path)
            self.feature_names = list(self.model.feature_name_)
            print("✅ PredictionHandler fully initialized with all MLOps artifacts.")
            
        except Exception as e:
            print(f"CRITICAL ERROR: Failed to load MLOps artifacts: {e}")
            self.model = None

        
    def _clean_names(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = df.columns
        new_cols = []
        for col in cols:
            new_col = re.sub(r'[^A-Za-z0-9_]+', '', col)
            new_cols.append(new_col)
        df.columns = new_cols
        return df


    def preprocess(self, input_df: pd.DataFrame) -> pd.DataFrame:
        
        df = input_df.copy()

        # 1. Anomaly Fix
        if 'DAYS_EMPLOYED' in df.columns and not df['DAYS_EMPLOYED'].isnull().all():
            DAYS_EMPLOYED_ANOMALY = self.imputation_maps.get('DAYS_EMPLOYED_ANOMALY', 365243)
            if df['DAYS_EMPLOYED'].eq(DAYS_EMPLOYED_ANOMALY).any():
                df['DAYS_EMPLOYED'].replace(DAYS_EMPLOYED_ANOMALY, np.nan, inplace=True)
            df['DAYS_EMPLOYED'] = np.abs(df['DAYS_EMPLOYED'])
        
        # 2. Ratio Features
        if 'AMT_CREDIT' in df.columns and 'AMT_INCOME_TOTAL' in df.columns:
            df['CREDIT_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
        if 'AMT_ANNUITY' in df.columns and 'AMT_INCOME_TOTAL' in df.columns:
            df['ANNUITY_INCOME_RATIO'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
        
        # 3. Time Feature Conversion
        if 'TIME_INDEX' in df.columns and df['TIME_INDEX'].notna().any():
            try:
                df['YEAR'] = pd.to_datetime(df['TIME_INDEX'].astype(str)).dt.year
                df = df.drop(columns=['TIME_INDEX'], errors='ignore') 
            except Exception as e:
                pass
                
        # B. Target Encoding
        df = self.target_encoder.transform(df)
        df = df.drop(columns=self.target_encoder.cols, errors='ignore') 
        
        # C. Final Cleaning and Alignment
        df = self._clean_names(df)

        # Handle NaN/Inf using SAVED MEANS
        df = df.replace([np.inf, -np.inf], np.nan)
        
        for col, mean_val in self.imputation_maps.items():
            if col in df.columns and col != 'DAYS_EMPLOYED_ANOMALY': 
                df[col] = df[col].fillna(mean_val)
        
        # Align Columns
        processed_df = df[[col for col in self.feature_names if col in df.columns]]
        processed_df = processed_df.reindex(columns=self.feature_names, fill_value=0) 
        
        return processed_df


    def predict_proba(self, raw_input_data: Dict[str, Any]) -> float:
        
        if self.model is None:
            return 0.5 

        # 1. Convert input dictionary/JSON to DataFrame
        input_df = pd.DataFrame([raw_input_data])
        
        # 2. Preprocess the data
        processed_df = self.preprocess(input_df)
        
        # 3. Generate prediction probability
        prediction_proba = self.model.predict_proba(processed_df)[0][1]
        
        return prediction_proba
