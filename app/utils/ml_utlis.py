import datetime
from typing import Dict
from fastapi import HTTPException

def predict_exoplanet(features: Dict[str, float], predictor):
    """Run ML model prediction for single exoplanet"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="ML model not loaded")
    
    result = predictor.predict_single(features)
    result['timestamp'] = datetime.now().isoformat()

    if result.get('error'):
        raise HTTPException(status_code=400, detail=result['error'])
    
    return result
