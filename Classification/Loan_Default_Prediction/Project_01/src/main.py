# FILE: src/main.py

from fastapi import FastAPI, HTTPException
import pandas as pd
import os
import sys

# -----------------------------------------
# Resolve project root dynamically
# -----------------------------------------
PROJECT_BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add 'src' directory to Python path
SRC_PATH = os.path.join(PROJECT_BASE_PATH, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)
    print("✅ Successfully added 'src' directory to Python path.")

# -----------------------------------------
# Imports from project modules
# -----------------------------------------
from src.predict import PredictionHandler
from src.schemas import LoanApplicationRawInput, PredictionResponse

# -----------------------------------------
# Paths to model artifacts
# -----------------------------------------
MODEL_DIR = os.path.join(PROJECT_BASE_PATH, "models")

MODEL_PATH = os.path.join(MODEL_DIR, "final_lgbm_model.pkl")
IMPUTATION_PATH = os.path.join(MODEL_DIR, "final_imputation_map.json")
ENCODER_PATH = os.path.join(MODEL_DIR, "final_target_encoder.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "FINAL_MODEL_FEATURES.json")

# -----------------------------------------
# Load Prediction Handler ONCE at startup
# -----------------------------------------
prediction_handler = None
try:
    prediction_handler = PredictionHandler(
        model_path=MODEL_PATH,
        imputation_path=IMPUTATION_PATH,
        encoder_path=ENCODER_PATH,
        features_path=FEATURES_PATH
    )
    print("✅ PredictionHandler initialized successfully.")

except Exception as e:
    print(f"❌ CRITICAL ERROR: Failed to load model artifacts: {e}")

# -----------------------------------------
# FastAPI Setup
# -----------------------------------------
app = FastAPI(
    title="Home Credit Default Risk Predictor",
    description="Predicts the probability of loan default (TARGET=1) using the final LightGBM model.",
    version="1.0.0"
)

# -----------------------------------------
# Healthcheck Endpoint
# -----------------------------------------
@app.get("/health")
def health_check():
    if prediction_handler and hasattr(prediction_handler, "model") and prediction_handler.model:
        return {"status": "ok", "model_ready": True}
    else:
        raise HTTPException(
            status_code=503,
            detail="Service Unavailable: Model or artifacts failed to load."
        )

# -----------------------------------------
# Prediction Endpoint
# -----------------------------------------
@app.post("/predict", response_model=PredictionResponse)
def predict_loan_default(raw_input: LoanApplicationRawInput):
    """
    Receives raw loan application data, preprocesses according to model pipeline,
    and returns default probability.
    """

    if not prediction_handler or not hasattr(prediction_handler, "model") or not prediction_handler.model:
        raise HTTPException(status_code=503, detail="Prediction service not initialized.")

    # Convert input to dict
    raw_data = raw_input.model_dump()

    # Extract SK_ID before preprocessing
    sk_id = raw_data.pop("SK_ID_CURR")

    try:
        probability = prediction_handler.predict_proba(raw_data)

    except Exception as e:
        print(f"Prediction Error for SK_ID {sk_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal prediction failure.")

    return PredictionResponse(
        SK_ID_CURR=sk_id,
        probability_of_default=probability
    )
