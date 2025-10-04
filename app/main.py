from fastapi import Request, UploadFile, File
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import Any, Dict, Optional
from datetime import datetime
from dotenv import load_dotenv

# Load env
load_dotenv()

# Functions import
from .utils import (
    setup_gemini, create_app, chat_with_gemini, predict_exoplanet,
    load_predictor, predict_batch_from_csv, sample_features,
)

# Setup
model = setup_gemini()
app, templates = create_app()
predictor = load_predictor()

# Pydantic models
class ExoplanetFeatures(BaseModel):
    koi_ror: float = Field(..., ge=0.001, le=1.0)
    koi_impact: float = Field(..., ge=0.0, le=1.0)
    koi_depth: float = Field(..., ge=0.0, le=1000.0)
    koi_prad: float = Field(..., ge=0.1, le=50.0)
    koi_teq: float = Field(..., ge=100.0, le=5000.0)
    koi_duration: float = Field(..., ge=0.1, le=50.0)
    koi_insol: float = Field(..., ge=0.1, le=10000.0)
    koi_steff: float = Field(..., ge=2000.0, le=10000.0)

class PredictionResponse(BaseModel):
    classification: str
    confidence: float
    feature_importance: Optional[Dict[str, float]] = None
    timestamp: str
    error: Optional[str] = None

class ChatRequest(BaseModel):
    message: str
    context: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    reply: str
    timestamp: str
    context_used: bool = False

# Routes
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    reply = await chat_with_gemini(request.message, request.context, model)
    return ChatResponse(reply=reply, timestamp=datetime.now().isoformat(), context_used=bool(request.context))

@app.post("/api/predict/single", response_model=PredictionResponse)
async def predict_single(features: ExoplanetFeatures):
    result = predict_exoplanet(features.dict(), predictor)
    return PredictionResponse(**result)

@app.post("/api/predict/batch")
async def predict_batch(file: UploadFile = File(...)):
    if not file.filename.lower().endswith('.csv'):
        return {"detail": "Please upload a CSV file."}
    content = await file.read()
    return predict_batch_from_csv(content, predictor)

@app.get("/api/sample/data")
async def get_sample_data():
    return sample_features()
