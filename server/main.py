#!/usr/bin/env python3
"""
Cosmic Hunter FastAPI Backend
Main application entry point for the exoplanet detection API
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import sys
import os
import json
import asyncio
import io
from datetime import datetime
import sys, os

# Add the ml directory to the Python path

# Add the "ml" directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ml'))

from ml.exoplanet_utils import ExoplanetPredictor, create_sample_data


# Initialize FastAPI app
app = FastAPI(
    title="Cosmic Hunter API",
    description="AI-powered exoplanet detection and analysis API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global predictor instance
predictor = None

# Pydantic models for request/response
class ExoplanetFeatures(BaseModel):
    """Model for single exoplanet prediction request"""
    koi_ror: float = Field(..., description="Planet-to-star radius ratio", ge=0.001, le=1.0)
    koi_impact: float = Field(..., description="Impact parameter (0-1)", ge=0.0, le=1.0)
    koi_depth: float = Field(..., description="Transit depth in ppm", ge=0.0, le=1000.0)
    koi_prad: float = Field(..., description="Planetary radius in Earth radii", ge=0.1, le=50.0)
    koi_teq: float = Field(..., description="Equilibrium temperature in K", ge=100.0, le=5000.0)
    koi_duration: float = Field(..., description="Transit duration in hours", ge=0.1, le=50.0)
    koi_insol: float = Field(..., description="Insolation flux", ge=0.1, le=10000.0)
    koi_steff: float = Field(..., description="Stellar effective temperature in K", ge=2000.0, le=10000.0)

class PredictionResponse(BaseModel):
    """Model for prediction response"""
    classification: str
    confidence: float
    feature_importance: Optional[Dict[str, float]] = None
    timestamp: str
    error: Optional[str] = None

class BatchPredictionResponse(BaseModel):
    """Model for batch prediction response"""
    results: List[Dict[str, Any]]
    total_processed: int
    successful_predictions: int
    failed_predictions: int
    timestamp: str

class ChatRequest(BaseModel):
    """Model for chat request"""
    message: str
    context: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    """Model for chat response"""
    reply: str
    timestamp: str
    context_used: bool = False

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize the ML model on startup"""
    global predictor
    try:
        # Initialize predictor with model files
        model_path = os.path.join(os.path.dirname(__file__), '..', 'ml', 'exoplanet_model.pkl')
        scaler_path = os.path.join(os.path.dirname(__file__), '..', 'ml', 'scaler.pkl')
        
        predictor = ExoplanetPredictor(model_path, scaler_path)
        print("✅ ML model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load ML model: {e}")
        raise

# Health check endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Cosmic Hunter API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "health": "/health",
            "predict_single": "/api/predict/single",
            "predict_batch": "/api/predict/batch",
            "chat": "/api/chat",
            "model_info": "/api/model/info",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="ML model not loaded")
    
    return {
        "status": "healthy",
        "model_loaded": True,
        "timestamp": datetime.now().isoformat()
    }

# Single prediction endpoint
@app.post("/api/predict/single", response_model=PredictionResponse)
async def predict_single(features: ExoplanetFeatures):
    """Make prediction for a single exoplanet candidate"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="ML model not loaded")
    
    try:
        # Convert Pydantic model to dict
        feature_dict = features.dict()
        
        # Make prediction
        result = predictor.predict_single(feature_dict)
        
        # Add timestamp
        result['timestamp'] = datetime.now().isoformat()
        
        # Handle errors
        if result.get('error'):
            raise HTTPException(status_code=400, detail=result['error'])
        
        return PredictionResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Batch prediction endpoint
@app.post("/api/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(file: UploadFile = File(...)):
    """Make predictions for a batch of exoplanet candidates from CSV file"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="ML model not loaded")
    
    try:
        # Read CSV file
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # Validate required columns
        required_columns = [
            'koi_ror', 'koi_impact', 'koi_depth', 'koi_prad',
            'koi_teq', 'koi_duration', 'koi_insol', 'koi_steff'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise HTTPException(
                status_code=400, 
                detail=f"Missing required columns: {missing_columns}"
            )
        
        # Convert DataFrame to list of dicts
        candidates = df[required_columns].to_dict('records')
        
        # Make predictions
        results = predictor.predict_batch(candidates)
        
        # Calculate statistics
        successful = sum(1 for r in results if r.get('error') is None)
        failed = len(results) - successful
        
        return BatchPredictionResponse(
            results=results,
            total_processed=len(results),
            successful_predictions=successful,
            failed_predictions=failed,
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

# Chat endpoint with Gemini AI integration
@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat with AI assistant about exoplanets using Gemini AI"""
    try:
        # Import Gemini service
        from gemini_service import get_gemini_response
        
        # Get response from Gemini AI
        reply = get_gemini_response(request.message, request.context)
        
        return ChatResponse(
            reply=reply,
            timestamp=datetime.now().isoformat(),
            context_used=request.context is not None
        )
        
    except Exception as e:
        print(f"❌ Chat error: {e}")
        # Fallback response
        return ChatResponse(
            reply="I'm experiencing some technical difficulties. Please try again in a moment, or feel free to ask about exoplanet science!",
            timestamp=datetime.now().isoformat(),
            context_used=False
        )

# Model information endpoint
@app.get("/api/model/info")
async def get_model_info():
    """Get information about the trained model"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="ML model not loaded")
    
    try:
        info = predictor.get_model_info()
        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}")

# Feature descriptions endpoint
@app.get("/api/features/descriptions")
async def get_feature_descriptions():
    """Get descriptions of model features"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="ML model not loaded")
    
    return predictor.get_feature_descriptions()

# Sample data endpoint
@app.get("/api/sample/data")
async def get_sample_data():
    """Get sample data for testing the API"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="ML model not loaded")
    
    return create_sample_data()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
