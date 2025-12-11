# FILE: src/main.py

from fastapi import FastAPI, HTTPException
import pandas as pd
import os

# Assuming you have a config module or paths defined globally
# from config import cfg 
# from src.predict import PredictionHandler
# from src.schemas import LoanApplicationRawInput, PredictionResponse


# --- MOCK CONFIGURATION (Replace with your actual config paths) ---
class MockConfig:
    MODEL_DIR = './models' 
cfg = MockConfig()

# Define artifact paths (Ensure these match your Block 19/20 saving paths)
MODEL_PATH = os.path.join(cfg.MODEL_DIR, 'final_lgbm_optimized_model.pkl')
MEANS_MAP_PATH = os.path.join(cfg.MODEL_DIR, 'imputation_means_map.pkl')
ENCODER_PATH = os.path.join(cfg.MODEL_DIR, 'final_target_encoder.pkl')

# --- APPLICATION LIFECYCLE: Load Model ONCE ---

# Initialize the Prediction Handler globally when the API server starts
# NOTE: You must replace the mock class below with your actual PredictionHandler
class MockPredictionHandler:
    def __init__(self, *args, **kwargs):
        print("MOCK Handler: Initializing and loading models...")
    def predict_proba(self, raw_data_dict):
        # MOCK LOGIC: Returns a random but plausible probability for testing
        return 0.15 
prediction_handler = MockPredictionHandler(MODEL_PATH, MEANS_MAP_PATH, ENCODER_PATH)


# --- FastAPI App Setup ---
app = FastAPI(
    title="Home Credit Default Risk Predictor",
    description="Predicts the probability of loan default (Target=1) using LightGBM.",
    version="1.0.0"
)

@app.get("/health")
def health_check():
    """Simple health check endpoint to confirm the service is running."""
    # In a real app, you'd check if prediction_handler.model is not None
    return {"status": "ok", "model_ready": True} 


@app.post("/predict", response_model=PredictionResponse)
def predict_loan_default(raw_input: LoanApplicationRawInput):
    """
    Receives raw customer data, preprocesses it using saved artifacts, 
    and returns the probability of default.
    """
    
    # 1. Convert Pydantic model data to a simple dictionary
    # FastAPI/Pydantic ensures the raw_input data is valid and correctly typed
    raw_data_dict = raw_input.model_dump() 
    
    # 2. Get the ID before processing
    sk_id = raw_data_dict.pop('SK_ID_CURR') # Pop ID, as it's not a feature

    try:
        # 3. Use the PredictionHandler to get the score
        probability = prediction_handler.predict_proba(raw_data_dict) 
        
    except Exception as e:
        # Catch any errors in feature engineering or prediction
        print(f"Prediction Error: {e}")
        raise HTTPException(status_code=500, detail="Internal prediction failure.")

    # 4. Return the structured response
    return PredictionResponse(
        SK_ID_CURR=sk_id,
        probability_of_default=probability
    )