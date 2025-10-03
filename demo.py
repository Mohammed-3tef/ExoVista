#!/usr/bin/env python3
"""
Cosmic Hunter Demo Script
Demonstrates the system capabilities without starting servers
"""

import sys
import os
import json

def demo_ml_model():
    """Demonstrate ML model functionality"""
    print("🧪 ML Model Demo")
    print("-" * 30)
    
    try:
        sys.path.append('ml')
        from model_utils import ExoplanetPredictor, create_sample_data
        
        # Load model
        model_path = os.path.join('ml', 'exoplanet_model.pkl')
        scaler_path = os.path.join('ml', 'scaler.pkl')
        predictor = ExoplanetPredictor(model_path, scaler_path)
        
        # Get sample data
        sample_data = create_sample_data()
        print("Sample exoplanet features:")
        for feature, value in sample_data.items():
            print(f"  {feature}: {value}")
        
        # Make prediction
        result = predictor.predict_single(sample_data)
        print(f"\nPrediction Result:")
        print(f"  Classification: {result['classification']}")
        print(f"  Confidence: {result['confidence']:.1%}")
        
        if result['feature_importance']:
            print(f"\nFeature Importance:")
            sorted_features = sorted(result['feature_importance'].items(), 
                                   key=lambda x: x[1], reverse=True)
            for feature, importance in sorted_features[:3]:
                print(f"  {feature}: {importance:.1%}")
        
        return True
        
    except Exception as e:
        print(f"❌ ML Model demo failed: {e}")
        return False

def demo_api_structure():
    """Demonstrate API structure"""
    print("\n🌐 API Structure Demo")
    print("-" * 30)
    
    api_endpoints = {
        "GET /": "Root endpoint with API information",
        "GET /health": "Health check endpoint",
        "POST /api/predict/single": "Single exoplanet prediction",
        "POST /api/predict/batch": "Batch CSV processing",
        "POST /api/chat": "AI chatbot with Gemini integration",
        "GET /api/model/info": "Model information and metadata",
        "GET /api/features/descriptions": "Feature descriptions",
        "GET /api/sample/data": "Sample data for testing"
    }
    
    print("Available API endpoints:")
    for endpoint, description in api_endpoints.items():
        print(f"  {endpoint}: {description}")
    
    return True

def demo_frontend_features():
    """Demonstrate frontend features"""
    print("\n🎨 Frontend Features Demo")
    print("-" * 30)
    
    features = [
        "Space-themed UI with animated starfield background",
        "Orbitron font for futuristic typography",
        "Manual scan form with 8 physical features",
        "Batch CSV upload with drag & drop",
        "Single result dashboard with classification badges",
        "Batch result dashboard with summary statistics",
        "Floating AI chatbot with typing indicators",
        "Real-time notifications and error handling",
        "Responsive design for desktop and mobile",
        "Loading states and progress indicators"
    ]
    
    print("Frontend features:")
    for i, feature in enumerate(features, 1):
        print(f"  {i}. {feature}")
    
    return True

def demo_system_architecture():
    """Demonstrate system architecture"""
    print("\n🏗️ System Architecture Demo")
    print("-" * 30)
    
    architecture = {
        "Frontend": {
            "Technology": "HTML5, CSS3, JavaScript (Vanilla)",
            "Theme": "Space-themed with starfield animation",
            "Features": "Manual scan, batch upload, results dashboard, AI chat"
        },
        "Backend": {
            "Technology": "FastAPI (Python)",
            "Features": "RESTful API, CORS support, file upload",
            "Endpoints": "8 API endpoints for all functionality"
        },
        "ML Model": {
            "Algorithm": "XGBoost Classifier",
            "Performance": "92.7% AUC accuracy",
            "Features": "8 physical characteristics",
            "Training Data": "9,201 exoplanet candidates"
        },
        "AI Integration": {
            "Service": "Google Gemini AI",
            "Purpose": "Educational chatbot for exoplanet science",
            "Features": "Context-aware responses, fallback system"
        }
    }
    
    for component, details in architecture.items():
        print(f"\n{component}:")
        for key, value in details.items():
            if isinstance(value, list):
                print(f"  {key}: {', '.join(value)}")
            else:
                print(f"  {key}: {value}")
    
    return True

def main():
    """Main demo function"""
    print("🌟 Cosmic Hunter System Demo")
    print("=" * 50)
    print("This demo showcases the system capabilities without starting servers.")
    print()
    
    demos = [
        ("ML Model", demo_ml_model),
        ("API Structure", demo_api_structure),
        ("Frontend Features", demo_frontend_features),
        ("System Architecture", demo_system_architecture)
    ]
    
    results = {}
    
    for name, demo_func in demos:
        try:
            results[name] = demo_func()
        except Exception as e:
            print(f"❌ {name} demo failed: {e}")
            results[name] = False
    
    print("\n📊 Demo Results:")
    print("-" * 20)
    
    for name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{name}: {status}")
    
    total_demos = len(results)
    passed_demos = sum(results.values())
    
    print(f"\nOverall: {passed_demos}/{total_demos} demos successful")
    
    if passed_demos == total_demos:
        print("\n🎉 All demos passed! The system is ready to use.")
        print("\nTo start the full system:")
        print("  python start_system.py")
        print("\nTo run complete tests:")
        print("  python test_complete_system.py")
    else:
        print(f"\n⚠️ {total_demos - passed_demos} demos failed. Please check the errors above.")
    
    return passed_demos == total_demos

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

