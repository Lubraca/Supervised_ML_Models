# FILE: src/predict.py

import pandas as pd
import numpy as np
import joblib
import json
import re
from typing import Any, Dict
import category_encoders as ce


class PredictionHandler:
    """
    Production-ready prediction pipeline for Home Credit Default Risk.
    Responsibilities:
    - Load LightGBM final model
    - Load target encoder
    - Load imputation map (.json)
    - Load final feature list (12 features)
    - Reproduce feature engineering from training pipeline
    """

    def __init__(self, model_path: str, imputation_path: str, encoder_path: str, features_path: str):
        try:
            # ----------------------------
            # Load final model (LightGBM)
            # ----------------------------
            self.model = joblib.load(model_path)

            # ----------------------------
            # Load imputation map (.json or .pkl)
            # ----------------------------
            if imputation_path.endswith(".json"):
                with open(imputation_path, "r") as f:
                    self.imputation_map = json.load(f)
            else:
                self.imputation_map = joblib.load(imputation_path)

            # ----------------------------
            # Load target encoder
            # ----------------------------
            self.target_encoder = joblib.load(encoder_path)

            # ----------------------------
            # Load final feature list (12 features)
            # ----------------------------
            with open(features_path, "r") as f:
                self.final_features = json.load(f)

            self.expected_feature_count = len(self.final_features)

            print(f"[INIT] Model ready. Using {self.expected_feature_count} final features.")

        except Exception as e:
            print(f"[CRITICAL] Failed to load model artifacts: {e}")
            raise

    # ============================================================
    # Helper: clean column names consistently
    # ============================================================
    @staticmethod
    def _clean_single_name(name: str) -> str:
        return re.sub(r"[^A-Za-z0-9_]+", "", str(name))

    def _clean_names(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = [self._clean_single_name(c) for c in df.columns]
        return df

    # ============================================================
    # Feature Engineering (same logic as Notebook 03)
    # ============================================================
    def _feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # DAYS_EMPLOYED anomaly fix
        if "DAYS_EMPLOYED" in df.columns:
            anomaly_flag = 365243
            df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(anomaly_flag, np.nan)
            df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].abs()

        # Ratio: CREDIT / INCOME
        if "AMT_CREDIT" in df.columns and "AMT_INCOME_TOTAL" in df.columns:
            df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]

        # Ratio: ANNUITY / INCOME
        if "AMT_ANNUITY" in df.columns and "AMT_INCOME_TOTAL" in df.columns:
            df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]

        # PAYMENT_RATE = ANNUITY / CREDIT
        if "AMT_ANNUITY" in df.columns and "AMT_CREDIT" in df.columns:
            df["PAYMENT_RATE"] = df["AMT_ANNUITY"] / df["AMT_CREDIT"]

        return df

    # ============================================================
    # Target Encoding (same as Notebook 03)
    # ============================================================
    def _apply_target_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # If encoder stores .cols (categorical columns)
        if hasattr(self.target_encoder, "cols"):
            cat_cols = [c for c in self.target_encoder.cols if c in df.columns]
            if not cat_cols:
                return df

            df_cat = df[cat_cols].copy()
            df_num = df.drop(columns=cat_cols, errors="ignore")

            df_cat_enc = self.target_encoder.transform(df_cat)
            # Match training-time naming: <col>_TARGET_ENC
            df_cat_enc.columns = [f"{c}_TARGET_ENC" for c in df_cat_enc.columns]
            df_num = df_num.drop(columns=df_cat_enc.columns, errors="ignore")
            df = pd.concat([df_num, df_cat_enc], axis=1)
        else:
            # generic fallback
            df = self.target_encoder.transform(df)
            if isinstance(df, pd.DataFrame):
                df.columns = [f"{c}_TARGET_ENC" for c in df.columns]

        return df

    # ============================================================
    # Preprocessing Pipeline (Core)
    # ============================================================
    def preprocess(self, raw_input: Dict[str, Any]) -> pd.DataFrame:
        df = pd.DataFrame([raw_input])

        # 1. Clean names
        df = self._clean_names(df)

        # 2. Feature Engineering
        df = self._feature_engineering(df)

        # 3. Target Encoding
        df = self._apply_target_encoding(df)

        # 4. Clean names again (encoders may produce unexpected labels)
        df = self._clean_names(df)

        # 5. Align to final model features
        df = df.reindex(columns=self.final_features, fill_value=np.nan)

        # 6. Imputation (mean)
        for col, mean_val in self.imputation_map.items():
            if col in df.columns:
                df[col] = df[col].fillna(mean_val)

        return df

    # ============================================================
    # Predict Probability
    # ============================================================
    def predict_proba(self, raw_input: Dict[str, Any]) -> float:
        processed = self.preprocess(raw_input)
        proba = float(self.model.predict_proba(processed)[0][1])
        return proba
