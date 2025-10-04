import datetime
from typing import Dict, Any, List, Tuple
from fastapi import HTTPException
import io
import csv
import os
import joblib
import pandas as pd

# ============================================================
# Heuristic Fallback Predictor
# ============================================================
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
        value = max(lo, min(value, hi))  # clip
        if key == 'koi_impact':  # inverse correlation
            value = hi - (value - lo)
        return (value - lo) / (hi - lo) if hi > lo else 0.0

    def _score(self, features: Dict[str, float]) -> float:
        score, total_weight = 0.0, 0.0
        for key, weight in self.feature_weights.items():
            if key in features and features[key] != 0.0:
                norm = self._normalize(key, float(features.get(key, 0.0)))
                score += weight * norm
                total_weight += weight
        if total_weight > 0:
            score /= total_weight

        # Boost for Earth-like
        if 0.8 <= features.get('koi_prad', 0) <= 2.0:
            score += 0.15
        if 200 <= features.get('koi_teq', 0) <= 350:
            score += 0.15

        return min(max(score, 0.0), 1.0)

    def predict_single(self, features: Dict[str, float]) -> Dict[str, Any]:
        try:
            confidence = self._score(features)
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


# ============================================================
# Trained Model Predictor
# ============================================================
class TrainedExoplanetPredictor:
    def __init__(self, model_path: str, scaler_path: str, threshold_path: str, feature_names_path: str):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.threshold = joblib.load(threshold_path)
        self.feature_names = joblib.load(feature_names_path)

    def predict_single(self, features: Dict[str, float]) -> Dict[str, Any]:
        try:
            df = pd.DataFrame([features])
            # Add missing columns with 0.0
            for col in self.feature_names:
                if col not in df.columns:
                    df[col] = 0.0
            # Ensure same column order
            df = df[self.feature_names]

            X_scaled = self.scaler.transform(df)
            y_proba = self.model.predict_proba(X_scaled)[:, 1]
            confidence = float(y_proba[0])
            classification = "CONFIRMED" if confidence >= self.threshold else "FALSE POSITIVE"

            return {
                "classification": classification,
                "confidence": confidence,
                "threshold": float(self.threshold),
            }
        except Exception as e:
            return {
                "classification": "ERROR",
                "confidence": 0.0,
                "error": f"Trained prediction error: {e}",
            }


# ============================================================
# Loader: choose trained model if available, fallback otherwise
# ============================================================
def load_predictor():
    model_path = "app/models/saved/lgbm_model.pkl"
    scaler_path = "app/models/saved/scaler.pkl"
    threshold_path = "app/models/saved/threshold.pkl"
    feature_names_path = "app/models/saved/feature_names.pkl"

    if all(os.path.exists(p) for p in [model_path, scaler_path, threshold_path, feature_names_path]):
        return TrainedExoplanetPredictor(model_path, scaler_path, threshold_path, feature_names_path)
    else:
        return ExoplanetPredictor()  # fallback heuristic


# ============================================================
# Prediction helpers
# ============================================================
def predict_exoplanet(features: Dict[str, float], predictor) -> Dict[str, Any]:
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


def predict_batch_from_csv(file_bytes: bytes, predictor) -> Dict[str, Any]:
    if predictor is None:
        raise HTTPException(status_code=503, detail="ML model not loaded")
    headers, rows = _parse_csv_bytes(file_bytes)

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
