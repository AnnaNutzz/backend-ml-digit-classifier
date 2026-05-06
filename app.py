from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import os
import subprocess

# Initialize FastAPI
app = FastAPI()

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model.pkl on startup
MODEL_PATH = "model.pkl"
model = None

@app.on_event("startup")
def load_model():
    global model
    # If model.pkl does not exist, train and save it automatically
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Running training script...")
        try:
            subprocess.run(["python3", "model_train.py"], check=True)
        except Exception as e:
            print(f"Error during training: {e}")
            raise RuntimeError("Could not train model.")
    
    print("Loading model...")
    model = joblib.load(MODEL_PATH)

class PredictionInput(BaseModel):
    # Expecting 784 flattened pixel values (28x28)
    image_data: list

@app.post("/predict")
async def predict(data: PredictionInput):
    """
    Accepts input as JSON body.
    Returns prediction and confidence.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    
    try:
        # Convert input to numpy array
        input_array = np.array(data.image_data).reshape(1, -1)
        
        # Get prediction
        prediction = model.predict(input_array)[0]
        
        # Get confidence (using proba if available)
        probabilities = model.predict_proba(input_array)[0]
        confidence = float(np.max(probabilities))
        
        return {
            "prediction": str(prediction),
            "confidence": confidence
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Use port from environment or default to 8000
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
