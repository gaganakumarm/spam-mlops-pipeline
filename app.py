from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from src.inference import InferencePipeline

# 1. Initialize the FastAPI app
app = FastAPI(
    title="Spam Detection API",
    description="MLOps Pipeline Demo: Real-time Spam Classification",
    version="0.1.0"
)

# 2. Define the Request Data structure
class RequestData(BaseModel):
    text: str

# 3. Initialize the Inference Pipeline
pipeline = InferencePipeline(
    model_uri=None,
    vectorizer_uri=None,
    local_model_path="artifacts/model/model.pkl",
    local_vectorizer_path="artifacts/vectorizer/vectorizer.pkl"
)

# 4. Define the Prediction Endpoint
@app.post("/predict")
async def predict(data: RequestData):
    try:
        # Call the predict method from our InferencePipeline
        prediction_label = pipeline.predict(data.text)
        
        return {
            "text": data.text,
            "label": prediction_label
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 5. Health Check Endpoint
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": pipeline.model is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)