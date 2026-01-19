from fastapi import FastAPI, HTTPException
import os

from src.predict import PredictionHandler
from src.schemas import LoanApplicationRawInput, PredictionResponse

# =====================================================
# PATH CONFIG (DOCKER / PRODUCTION)
# =====================================================

# Base path inside the container (override via env if needed)
BASE_DIR = os.getenv("PROJECT_BASE_PATH", "/app")

# Directory where the serialized artifacts live
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Artifact file paths
MODEL_PATH = os.path.join(MODEL_DIR, "final_lgbm_optimized_model.pkl")
MEANS_MAP_PATH = os.path.join(MODEL_DIR, "imputation_means_map.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "final_target_encoder.pkl")

# =====================================================
# MODEL INITIALIZATION (LOAD ONCE)
# =====================================================

prediction_handler = None

try:
    prediction_handler = PredictionHandler(
        model_path=MODEL_PATH,
        mean_map_path=MEANS_MAP_PATH,
        encoder_path=ENCODER_PATH
    )
    print("✅ PredictionHandler loaded successfully.")
except Exception as e:
    # Keep the API up, but mark the service as unavailable via /health
    print(f"❌ CRITICAL ERROR while loading artifacts: {e}")

# =====================================================
# FASTAPI APP
# =====================================================

app = FastAPI(
    title="Home Credit Default Risk API",
    description="Predicts the probability of loan default using LightGBM.",
    version="1.0.0"
)

# =====================================================
# ENDPOINTS
# =====================================================

@app.get("/health")
def health_check():
    """
    Health endpoint to confirm the service is running and the model is loaded.
    """
    if prediction_handler and getattr(prediction_handler, "model", None):
        return {"status": "ok", "model_loaded": True}

    raise HTTPException(
        status_code=503,
        detail="Service Unavailable: model/artifacts failed to load."
    )


@app.post("/predict", response_model=PredictionResponse)
def predict_loan_default(payload: LoanApplicationRawInput):
    """
    Receives raw customer data and returns the probability of default.
    """
    if not prediction_handler or not getattr(prediction_handler, "model", None):
        raise HTTPException(
            status_code=503,
            detail="Prediction service not initialized. Check server logs."
        )

    # Convert the Pydantic payload to a plain dict
    data = payload.model_dump()

    # Extract the identifier (not used as a model feature)
    sk_id = data.pop("SK_ID_CURR")

    try:
        # Run preprocessing + model inference
        probability = prediction_handler.predict_proba(data)
    except Exception as e:
        # Log the root cause server-side, return a safe API error to the client
        print(f"❌ Prediction error for SK_ID_CURR={sk_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal prediction failure."
        )

    # Return a structured response
    return PredictionResponse(
        SK_ID_CURR=sk_id,
        probability_of_default=probability
    )
