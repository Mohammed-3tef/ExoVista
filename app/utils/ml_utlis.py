
import datetime
from typing import Dict, Any, List, Tuple
from fastapi import HTTPException
import io
import csv

class ExoplanetPredictor:
    """
    Minimal heuristic predictor as a fallback until a trained model is wired.
    Produces a confidence score in [0, 1] and a simple classification.
    """
    def __init__(self):
        # Feature weights roughly reflecting domain intuition
        self.feature_weights = {
            'koi_prad': 0.23,
            'koi_insol': 0.16,
            'koi_ror': 0.15,
            'koi_teq': 0.12,
            'koi_depth': 0.12,
            'koi_steff': 0.10,
            'koi_duration': 0.07,
            'koi_impact': 0.05,
        }

    def _normalize(self, key: str, value: float) -> float:
        # Min/max clips based on UI ranges
        ranges = {
            'koi_ror': (0.001, 1.0),
            'koi_impact': (0.0, 1.0),
            'koi_depth': (0.0, 1000.0),
            'koi_prad': (0.1, 50.0),
            'koi_teq': (100.0, 5000.0),
            'koi_duration': (0.1, 50.0),
            'koi_insol': (0.1, 10000.0),
            'koi_steff': (2000.0, 10000.0),
        }
        lo, hi = ranges.get(key, (0.0, 1.0))
        if value is None:
            return 0.0
        if value < lo:
            value = lo
        if value > hi:
            value = hi
        # Some features correlate inversely; lightly adjust
        invert = {'koi_impact'}
        if key in invert:
            value = hi - (value - lo)
        return (value - lo) / (hi - lo) if hi > lo else 0.0

    def _score(self, features: Dict[str, float]) -> float:
        score = 0.0
        total_weight = 0.0
        
        # Calculate weighted score based on available features
        for key, weight in self.feature_weights.items():
            if key in features and features[key] != 0.0:
                normalized_value = self._normalize(key, float(features.get(key, 0.0)))
                score += weight * normalized_value
                total_weight += weight
        
        # Adjust score if we have partial features
        if total_weight > 0:
            score = score / total_weight
            
        # Boost scores for Earth-like planets
        if 'koi_prad' in features and 0.8 <= features['koi_prad'] <= 2.0:
            score += 0.15
        if 'koi_teq' in features and 200 <= features['koi_teq'] <= 350:
            score += 0.15
            
        # Clip to [0,1]
        if score < 0.0:
            score = 0.0
        if score > 1.0:
            score = 1.0
        return score

    def predict_single(self, features: Dict[str, float]) -> Dict[str, Any]:
        try:
            confidence = self._score(features)
            
            # Three-category classification based on confidence thresholds
            if confidence >= 0.7:
                classification = 'CONFIRMED'
            elif confidence >= 0.4:
                classification = 'CANDIDATE'
            else:
                classification = 'FALSE POSITIVE'
                
            return {
                'classification': classification,
                'confidence': float(confidence),
                'feature_importance': self.feature_weights,
            }
        except Exception as e:
            return {
                'classification': 'ERROR',
                'confidence': 0.0,
                'feature_importance': None,
                'error': f'Prediction error: {e}',
            }


def load_predictor() -> ExoplanetPredictor:
    """Load the trained predictor; fall back to heuristic predictor if unavailable."""
    # Placeholder: if you later persist a trained model, load it here.
    return ExoplanetPredictor()


def predict_exoplanet(features: Dict[str, float], predictor: ExoplanetPredictor) -> Dict[str, Any]:
    """Run ML model prediction for single exoplanet"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="ML model not loaded")
    result = predictor.predict_single(features)
    result['timestamp'] = datetime.datetime.now().isoformat()
    if result.get('error'):
        raise HTTPException(status_code=400, detail=result['error'])
    return result


def _parse_csv_bytes(file_bytes: bytes) -> Tuple[List[str], List[Dict[str, float]]]:
    text = file_bytes.decode('utf-8', errors='ignore')
    reader = csv.DictReader(io.StringIO(text))
    rows: List[Dict[str, float]] = []
    headers = reader.fieldnames or []
    for row in reader:
        clean: Dict[str, float] = {}
        for k, v in row.items():
            if k is None:
                continue
            try:
                clean[k] = float(v) if v not in (None, '', 'NaN') else 0.0
            except Exception:
                clean[k] = 0.0
        rows.append(clean)
    return headers, rows


def predict_batch_from_csv(file_bytes: bytes, predictor: ExoplanetPredictor) -> Dict[str, Any]:
    if predictor is None:
        raise HTTPException(status_code=503, detail="ML model not loaded")
    headers, rows = _parse_csv_bytes(file_bytes)
    required = [
        'koi_ror', 'koi_impact', 'koi_depth', 'koi_prad',
        'koi_teq', 'koi_duration', 'koi_insol', 'koi_steff',
    ]
    missing = [c for c in required if c not in headers]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing required columns: {', '.join(missing)}")

    results: List[Dict[str, Any]] = []
    for r in rows:
        res = predictor.predict_single(r)
        res['timestamp'] = datetime.datetime.now().isoformat()
        results.append(res)

    successful = len([r for r in results if not r.get('error')])
    confirmed = len([r for r in results if r.get('classification') == 'CONFIRMED'])
    return {
        'total_processed': len(results),
        'successful_predictions': successful,
        'confirmed_count': confirmed,
        'results': results,
    }


def sample_features() -> Dict[str, float]:
    return {
        'koi_ror': 0.12,
        'koi_impact': 0.25,
        'koi_depth': 250.0,
        'koi_prad': 2.1,
        'koi_teq': 800.0,
        'koi_duration': 5.4,
        'koi_insol': 900.0,
        'koi_steff': 5800.0,
    }
 
