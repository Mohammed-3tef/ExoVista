#!/usr/bin/env python3
"""
Cosmic Hunter Model Utilities
Preprocessing and inference functions for the exoplanet detection model
"""

import pandas as pd
import numpy as np
import joblib
import os
from typing import Dict, List, Tuple, Any


# Required features for the model
REQUIRED_FEATURES = [
    'koi_ror',      # Planet-to-star radius ratio
    'koi_impact',   # Impact parameter
    'koi_depth',    # Transit depth
    'koi_prad',     # Planetary radius
    'koi_teq',      # Equilibrium temperature
    'koi_duration', # Transit duration
    'koi_insol',    # Insolation flux
    'koi_steff'     # Stellar effective temperature
]

class ExoplanetPredictor:
    """Exoplanet prediction model wrapper"""
    
    def __init__(self, model_path: str = 'exoplanet_model.pkl', scaler_path: str = 'scaler.pkl'):
        """Initialize the predictor with trained model and scaler"""
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None
        self.feature_names = REQUIRED_FEATURES
        self.load_model()
    
    def load_model(self):
        """Load the trained model and scaler"""
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                print(f"✅ Model loaded from {self.model_path}")
            else:
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            if os.path.exists(self.scaler_path):
                self.scaler = joblib.load(self.scaler_path)
                print(f"✅ Scaler loaded from {self.scaler_path}")
            else:
                raise FileNotFoundError(f"Scaler file not found: {self.scaler_path}")
                
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def validate_input(self, data: Dict[str, float]) -> Tuple[bool, str]:
        """Validate input data"""
        # Check if all required features are present
        missing_features = [f for f in self.feature_names if f not in data]
        if missing_features:
            return False, f"Missing features: {missing_features}"
        
        # Check for None or NaN values
        for feature, value in data.items():
            if value is None or (isinstance(value, float) and np.isnan(value)):
                return False, f"Invalid value for {feature}: {value}"
        
        # Check for reasonable ranges (basic validation)
        ranges = {
            'koi_ror': (0.001, 1.0),      # Radius ratio should be small
            'koi_impact': (0.0, 1.0),     # Impact parameter 0-1
            'koi_depth': (0.0, 1000.0),   # Transit depth in ppm
            'koi_prad': (0.1, 50.0),      # Planetary radius in Earth radii
            'koi_teq': (100.0, 5000.0),   # Equilibrium temperature in K
            'koi_duration': (0.1, 50.0),  # Transit duration in hours
            'koi_insol': (0.1, 10000.0),  # Insolation flux
            'koi_steff': (2000.0, 10000.0) # Stellar temperature in K
        }
        
        for feature, value in data.items():
            if feature in ranges:
                min_val, max_val = ranges[feature]
                if not (min_val <= value <= max_val):
                    return False, f"{feature} value {value} outside reasonable range [{min_val}, {max_val}]"
        
        return True, "Valid"
    
    def preprocess_single(self, data: Dict[str, float]) -> np.ndarray:
        """Preprocess single prediction input"""
        # Convert to DataFrame with single row
        df = pd.DataFrame([data])
        
        # Ensure correct feature order
        features = df[self.feature_names].values
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        return features_scaled
    
    def preprocess_batch(self, data: List[Dict[str, float]]) -> np.ndarray:
        """Preprocess batch prediction input"""
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Ensure correct feature order
        features = df[self.feature_names].values
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        return features_scaled
    
    def predict_single(self, data: Dict[str, float]) -> Dict[str, Any]:
        """Make prediction for single exoplanet candidate"""
        # Validate input
        is_valid, message = self.validate_input(data)
        if not is_valid:
            return {
                'error': message,
                'classification': None,
                'confidence': None,
                'feature_importance': None
            }
        
        try:
            # Preprocess
            features = self.preprocess_single(data)
            
            # Make prediction
            prediction_proba = self.model.predict_proba(features)[0]
            confidence = float(prediction_proba[1])  # Probability of being CONFIRMED
            
            # Classification
            classification = "CONFIRMED" if confidence > 0.5 else "NOT_CONFIRMED"
            
            # Feature importance (if available)
            feature_importance = None
            if hasattr(self.model, 'feature_importances_'):
                feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
            
            return {
                'classification': classification,
                'confidence': confidence,
                'feature_importance': feature_importance,
                'error': None
            }
            
        except Exception as e:
            return {
                'error': f"Prediction error: {str(e)}",
                'classification': None,
                'confidence': None,
                'feature_importance': None
            }
    
    def predict_batch(self, data: List[Dict[str, float]]) -> List[Dict[str, Any]]:
        """Make predictions for batch of exoplanet candidates"""
        results = []
        
        for i, candidate in enumerate(data):
            result = self.predict_single(candidate)
            result['id'] = i  # Add ID for tracking
            results.append(result)
        
        return results
    
    def get_feature_descriptions(self) -> Dict[str, str]:
        """Get descriptions of model features"""
        return {
            'koi_ror': 'Planet-to-star radius ratio',
            'koi_impact': 'Impact parameter (0-1, where 0 is center transit)',
            'koi_depth': 'Transit depth in parts per million',
            'koi_prad': 'Planetary radius in Earth radii',
            'koi_teq': 'Equilibrium temperature in Kelvin',
            'koi_duration': 'Transit duration in hours',
            'koi_insol': 'Insolation flux (stellar flux at planet)',
            'koi_steff': 'Stellar effective temperature in Kelvin'
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the trained model"""
        if self.model is None:
            return {'error': 'Model not loaded'}
        
        info = {
            'model_type': type(self.model).__name__,
            'features': self.feature_names,
            'feature_count': len(self.feature_names),
            'feature_descriptions': self.get_feature_descriptions()
        }
        
        # Add model-specific info if available
        if hasattr(self.model, 'n_estimators'):
            info['n_estimators'] = self.model.n_estimators
        if hasattr(self.model, 'max_depth'):
            info['max_depth'] = self.model.max_depth
        if hasattr(self.model, 'learning_rate'):
            info['learning_rate'] = self.model.learning_rate
        
        return info

def load_feature_importance(file_path: str = 'feature_importance.csv') -> pd.DataFrame:
    """Load feature importance from CSV file"""
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Feature importance file not found: {file_path}")
        return pd.DataFrame()

def create_sample_data() -> Dict[str, float]:
    """Create sample data for testing"""
    return {
        'koi_ror': 0.02,      # 2% radius ratio
        'koi_impact': 0.3,    # 30% impact parameter
        'koi_depth': 400.0,   # 400 ppm depth
        'koi_prad': 1.5,      # 1.5 Earth radii
        'koi_teq': 1200.0,    # 1200K temperature
        'koi_duration': 3.5,  # 3.5 hours duration
        'koi_insol': 1000.0,  # 1000 insolation
        'koi_steff': 5500.0   # 5500K stellar temperature
    }

def test_model():
    """Test the model with sample data"""
    print("🧪 Testing Exoplanet Predictor...")
    
    try:
        # Initialize predictor
        predictor = ExoplanetPredictor()
        
        # Get model info
        info = predictor.get_model_info()
        print(f"Model info: {info}")
        
        # Test with sample data
        sample_data = create_sample_data()
        result = predictor.predict_single(sample_data)
        
        print(f"Sample prediction: {result}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    test_model()

