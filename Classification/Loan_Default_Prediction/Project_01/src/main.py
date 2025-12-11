# FILE: src/main.py

from fastapi import FastAPI, HTTPException
import pandas as pd
import os
import sys

# Define project Path in Colab
PROJECT_BASE_PATH = '/content/drive/MyDrive/Project_01' 

# ADD 'src' DIRECTORY TO PYTHON PATH
SRC_PATH = os.path.join(PROJECT_BASE_PATH, 'src')

# verify if SRC_PATH is already in sys.path
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)
    print("✅ Successfully added 'src' directory to Python path.")

# Imports from your project directory:
from config import Paths
from src.predict import PredictionHandler
from src.schemas import LoanApplicationRawInput, PredictionResponse
    
cfg = Paths(PROJECT_BASE_PATH)
cfg.create_dirs() 

# Define artifact paths 
MODEL_PATH = os.path.join(cfg.MODEL_DIR, 'final_lgbm_optimized_model.pkl')
MEANS_MAP_PATH = os.path.join(cfg.MODEL_DIR, 'imputation_means_map.pkl')
ENCODER_PATH = os.path.join(cfg.MODEL_DIR, 'final_target_encoder.pkl')

# --- APPLICATION LIFECYCLE: Load Model ONCE ---

# Initialize the Prediction Handler globally when the API server starts
prediction_handler = None
try:
    prediction_handler = PredictionHandler(MODEL_PATH, MEANS_MAP_PATH, ENCODER_PATH)
    print("✅ PredictionHandler initialized successfully.")
    
except Exception as e:
    print(f"❌ CRITICAL ERROR: Failed to load model artifacts: {e}")


# --- FastAPI App Setup ---
app = FastAPI(
    title="Home Credit Default Risk Predictor",
    description="Predicts the probability of loan default (Target=1) using LightGBM.",
    version="1.0.0"
)

@app.get("/health")
def health_check():
    """Simple health check endpoint to confirm the service is running and model is loaded."""
    if prediction_handler and hasattr(prediction_handler, 'model') and prediction_handler.model:
        return {"status": "ok", "model_ready": True}
    else:
        # 503 Service Unavailable if the model failed to load at startup
        raise HTTPException(status_code=503, detail="Service Unavailable: Model or Artifacts failed to load.")


@app.post("/predict", response_model=PredictionResponse)
def predict_loan_default(raw_input: LoanApplicationRawInput):
    """
    Receives raw customer data, preprocesses it using saved artifacts, 
    and returns the probability of default.
    """
    if not prediction_handler or not hasattr(prediction_handler, 'model') or not prediction_handler.model:
        raise HTTPException(status_code=503, detail="Prediction service not initialized. Check server logs.")

    # 1. Convert Pydantic model data to a simple dictionary
    raw_data_dict = raw_input.model_dump() 
    
    # 2. Get the ID before processing
    sk_id = raw_data_dict.pop('SK_ID_CURR') 

    try:
        # 3. Use the PredictionHandler to get the score
        probability = prediction_handler.predict_proba(raw_data_dict) 
        
    except Exception as e:
        # Catch any errors in feature engineering or prediction
        print(f"Prediction Error for SK_ID {sk_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal prediction failure.")

    # 4. Return the structured response
    return PredictionResponse(
        SK_ID_CURR=sk_id,
        probability_of_default=probability
    )
