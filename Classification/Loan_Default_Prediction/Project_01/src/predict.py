
class PredictionHandler:
    """
    Handles loading MLOps artifacts and making consistent predictions 
    for single or batch inputs in a live environment.
    """
    
    # 1. Load All Artifacts in __init__
    def __init__(self, model_path: str, mean_map_path: str, encoder_path: str):
        
        try:
            # Load MLOps Artifacts
            self.model = joblib.load(model_path)
            self.imputation_maps = joblib.load(mean_map_path)   # Loads saved means/anomaly
            self.target_encoder = joblib.load(encoder_path)     # Loads saved encoder object
            
            # The list of feature names used during training is CRITICAL for alignment
            self.feature_names = list(self.model.feature_name_)
            print("✅ PredictionHandler fully initialized with all MLOps artifacts.")
            
        except Exception as e:
            print(f"CRITICAL ERROR: Failed to load MLOps artifacts: {e}")
            self.model = None

        
    def _clean_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Removes problematic characters for LightGBM/XGBoost compatibility."""
        cols = df.columns
        new_cols = []
        for col in cols:
            # Regex to keep only alphanumeric characters and underscores
            new_col = re.sub(r'[^A-Za-z0-9_]+', '', col)
            new_cols.append(new_col)
        df.columns = new_cols
        return df


    def preprocess(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """Applies the entire sequential preprocessing pipeline consistently."""
        
        df = input_df.copy()

        # --- A. (Feature Engineering) ---
        
        # 1. Anomaly Fix (Uses saved value from imputation map)
        if 'DAYS_EMPLOYED' in df.columns:
            # Load the saved value for the anomaly fix (365243)
            DAYS_EMPLOYED_ANOMALY = self.imputation_maps.get('DAYS_EMPLOYED_ANOMALY', 365243)
            df['DAYS_EMPLOYED'].replace(DAYS_EMPLOYED_ANOMALY, np.nan, inplace=True)
            df['DAYS_EMPLOYED'] = np.abs(df['DAYS_EMPLOYED'])
        
        # 2. Ratio Features (Must match training!)
        df['CREDIT_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
        df['ANNUITY_INCOME_RATIO'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
        
        # 3. Time Feature Conversion (Must match training!)
        if 'TIME_INDEX' in df.columns:
             try:
                 df['YEAR'] = pd.to_datetime(df['TIME_INDEX']).dt.year
                 df = df.drop(columns=['TIME_INDEX'])
             except:
                 pass
                
        # --- B. (Target Encoding) ---
        
        # Apply the SAVED encoder to the categorical columns of the live data
        df = self.target_encoder.transform(df)
        
        # Drop original categorical columns (they are now encoded)
        df = df.drop(columns=self.target_encoder.cols, errors='ignore') 
        
        # --- C. Final Cleaning and Alignment ---
        
        # 1. Clean Feature Names
        df = self._clean_names(df)

        # 2. Handle NaN/Inf using SAVED MEANS (CORRECT MLOPS IMPUTATION)
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Fill missing values using the means calculated from the TRAINING data
        for col, mean_val in self.imputation_maps.items():
            if col in df.columns and col != 'DAYS_EMPLOYED_ANOMALY': 
                df[col] = df[col].fillna(mean_val)
        
        # 3. Align Columns (CRITICAL MLOps Step)
        
        # Select and re-order columns to match the model's training list
        processed_df = df[[col for col in self.feature_names if col in df.columns]]
        # Fill any missing engineered features (that weren't in the raw input) with 0 or a consistent value
        processed_df = processed_df.reindex(columns=self.feature_names, fill_value=0) 
        
        return processed_df


    def predict_proba(self, raw_input_data: Dict[str, Any]) -> float:
        """
        Receives raw input data (e.g., from a JSON API request) and returns the 
        probability of default (Target=1).
        """
        if self.model is None:
            return 0.5 # Default prediction if model failed to load

        # 1. Convert input dictionary/JSON to DataFrame
        input_df = pd.DataFrame([raw_input_data])
        
        # 2. Preprocess the data
        processed_df = self.preprocess(input_df)
        
        # 3. Ensure column order matches the training data (CRITICAL!)
        # The reindex in preprocess should handle this, but an explicit check is safe:
        # processed_df = processed_df[self.feature_names] 

        # 4. Generate prediction probability
        prediction_proba = self.model.predict_proba(processed_df)[0][1]
        
        return prediction_proba