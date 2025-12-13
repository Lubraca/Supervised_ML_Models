# FILE: src/predict.py (Complete version)

import pandas as pd             
import joblib                   
import numpy as np              
import re                       
from typing import Dict, Any    
import category_encoders as ce

class PredictionHandler:
    
    def __init__(self, model_path: str, mean_map_path: str, encoder_path: str):
        """
        Initializes the prediction handler by loading the MLOps artifacts.
        """
        
        try:
            self.model = joblib.load(model_path)
            self.imputation_maps = joblib.load(mean_map_path) 
            self.target_encoder = joblib.load(encoder_path)
            
            # CRITICAL STEP: Read the feature names directly from the saved LightGBM model.
            # This ensures the API always aligns with the N features used during training.
            self.feature_names = [
                'PAYMENT_RATE', 'EXT_SOURCE_1','EXT_SOURCE_3','EXT_SOURCE_2','DAYS_BIRTH',
                'AMT_GOODS_PRICE', 'DAYS_EMPLOYED', 'AMT_ANNUITY', 'DAYS_ID_PUBLISH', 'AMT_CREDIT',
                'ORGANIZATION_TYPE_TARGET_ENC', 'NAME_EDUCATION_TYPE_TARGET_ENC', 'CODE_GENDER_TARGET_ENC',
                'OCCUPATION_TYPE_TARGET_ENC', 'ANNUITY_INCOME_RATIO', 'NAME_CONTRACT_TYPE_TARGET_ENC',
                'DAYS_LAST_PHONE_CHANGE','DAYS_REGISTRATION', 'NAME_FAMILY_STATUS_TARGET_ENC',
                'DEF_30_CNT_SOCIAL_CIRCLE', 'FLAG_OWN_CAR_TARGET_ENC', 'OWN_CAR_AGE',
                'FLAG_DOCUMENT_3', 'REGION_RATING_CLIENT_W_CITY', 'CREDIT_INCOME_RATIO'
                ]
            
            print(f"✅ PredictionHandler fully initialized with all MLOps artifacts. Model expects {len(self.feature_names)} features.")
            
        except Exception as e:
            print(f"❌ CRITICAL ERROR: Failed to load MLOps artifacts: {e}")
            self.model = None

        
    def _clean_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans feature names by removing special characters, necessary for compatibility 
        with various ML frameworks and environments.
        """
        cols = df.columns
        new_cols = []
        for col in cols:
            # Simple cleaning: replace non-alphanumeric/underscore with nothing
            new_col = re.sub(r'[^A-Za-z0-9_]+', '', col)
            new_cols.append(new_col)
        df.columns = new_cols
        return df


    def preprocess(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies all necessary feature engineering and cleaning steps 
        to transform raw input into the model's expected format (N features).
        """
        
        df = input_df.copy()

        # A. Feature Engineering Replication (from 03_Feature_Eng.ipynb)
        
        # 1. Anomaly Fix (DAYS_EMPLOYED: 365243)
        if 'DAYS_EMPLOYED' in df.columns and not df['DAYS_EMPLOYED'].isnull().all():
            DAYS_EMPLOYED_ANOMALY = self.imputation_maps.get('DAYS_EMPLOYED_ANOMALY', 365243)
            if df['DAYS_EMPLOYED'].eq(DAYS_EMPLOYED_ANOMALY).any():
                df['DAYS_EMPLOYED'].replace(DAYS_EMPLOYED_ANOMALY, np.nan, inplace=True)
            df['DAYS_EMPLOYED'] = np.abs(df['DAYS_EMPLOYED'])
        
        # 2. Ratio Features (Calculated before feature selection)
        if 'AMT_CREDIT' in df.columns and 'AMT_INCOME_TOTAL' in df.columns:
            df['CREDIT_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
        if 'AMT_ANNUITY' in df.columns and 'AMT_INCOME_TOTAL' in df.columns:
            df['ANNUITY_INCOME_RATIO'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
        
        # 3. Time Feature Conversion (if relevant feature is present)
        if 'TIME_INDEX' in df.columns and df['TIME_INDEX'].notna().any():
            try:
                df['YEAR'] = pd.to_datetime(df['TIME_INDEX'].astype(str)).dt.year
                df = df.drop(columns=['TIME_INDEX'], errors='ignore') 
            except Exception as e:
                pass
                
        # B. Target Encoding Transformation
        
        # NOTE: The TargetEncoder must be applied BEFORE dropping original columns.
        original_categorical_cols = list(self.target_encoder.cols)
        
        # Transform the input using the saved encoder
        df = self.target_encoder.transform(df)
        
        # Drop the original categorical columns (since they are now encoded)
        df = df.drop(columns=original_categorical_cols, errors='ignore')
        
        # C. Final Cleaning and Alignment
        
        # Drop any remaining non-numeric columns (objects)
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        df = df.drop(columns=categorical_cols, errors='ignore')

        df = self._clean_names(df)

        # Handle NaN/Inf before imputation
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # CRITICAL STEP 1: Feature Filtering (Reducing 133+ columns to N features)
        df = df.filter(items=self.feature_names) 
        
        # CRITICAL STEP 2: Imputation using SAVED MEANS
        for col, mean_val in self.imputation_maps.items():
            # Skip the anomaly placeholder
            if col in df.columns and col != 'DAYS_EMPLOYED_ANOMALY': 
                df[col] = df[col].fillna(mean_val)
        
        # CRITICAL STEP 3: Final Column Alignment (Re-indexing to ensure correct order)
        processed_df = df.reindex(columns=self.feature_names, fill_value=0) 
        
        return processed_df


    def predict_proba(self, raw_input_data: Dict[str, Any]) -> float:
        """
        Takes raw data, preprocesses it, and returns the prediction probability.
        """
        
        if self.model is None:
            return 0.5 # Safe default return value if model load failed

        # 1. Convert input dictionary/JSON to DataFrame
        input_df = pd.DataFrame([raw_input_data])
        
        # 2. Preprocess the data
        processed_df = self.preprocess(input_df)
        
        # 3. Generate prediction probability
        prediction_proba = self.model.predict_proba(processed_df)[0][1]
        
        return prediction_proba