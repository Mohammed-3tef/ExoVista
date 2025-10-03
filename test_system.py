#!/usr/bin/env python3
"""
Test script for Cosmic Hunter system
"""

import sys
import os
import requests
import time
import subprocess
import threading

def test_ml_model():
    """Test the ML model directly"""
    print("🧪 Testing ML Model...")
    
    try:
        sys.path.append(os.path.join('ml'))
        from model_utils import ExoplanetPredictor, create_sample_data
        
        # Test model loading
        predictor = ExoplanetPredictor()
        print("✅ Model loaded successfully")
        
        # Test prediction
        sample_data = create_sample_data()
        result = predictor.predict_single(sample_data)
        print(f"✅ Sample prediction: {result['classification']} (confidence: {result['confidence']:.3f})")
        
        return True
    except Exception as e:
        print(f"❌ ML Model test failed: {e}")
        return False

def test_api_endpoints():
    """Test API endpoints"""
    print("\n🌐 Testing API Endpoints...")
    
    base_url = "http://localhost:8000"
    
    try:
        # Test health endpoint
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Health endpoint working")
        else:
            print(f"❌ Health endpoint failed: {response.status_code}")
            return False
        
        # Test model info
        response = requests.get(f"{base_url}/api/model/info", timeout=5)
        if response.status_code == 200:
            print("✅ Model info endpoint working")
        else:
            print(f"❌ Model info endpoint failed: {response.status_code}")
            return False
        
        # Test sample data
        response = requests.get(f"{base_url}/api/sample/data", timeout=5)
        if response.status_code == 200:
            print("✅ Sample data endpoint working")
        else:
            print(f"❌ Sample data endpoint failed: {response.status_code}")
            return False
        
        # Test single prediction
        sample_data = {
            "koi_ror": 0.02,
            "koi_impact": 0.3,
            "koi_depth": 400.0,
            "koi_prad": 1.5,
            "koi_teq": 1200.0,
            "koi_duration": 3.5,
            "koi_insol": 1000.0,
            "koi_steff": 5500.0
        }
        
        response = requests.post(
            f"{base_url}/api/predict/single",
            json=sample_data,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Single prediction working: {result['classification']} ({result['confidence']:.3f})")
        else:
            print(f"❌ Single prediction failed: {response.status_code}")
            return False
        
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API server. Make sure it's running on port 8000")
        return False
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False

def start_backend_server():
    """Start the backend server in a separate thread"""
    print("🚀 Starting backend server...")
    
    def run_server():
        os.chdir('server')
        subprocess.run([sys.executable, 'main.py'], capture_output=True)
    
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    
    # Wait for server to start
    time.sleep(3)
    return server_thread

def main():
    """Main test function"""
    print("🚀 Cosmic Hunter System Test")
    print("=" * 50)
    
    # Test ML model
    ml_success = test_ml_model()
    
    # Start backend server
    server_thread = start_backend_server()
    
    # Wait a bit more for server to fully start
    time.sleep(2)
    
    # Test API endpoints
    api_success = test_api_endpoints()
    
    # Summary
    print("\n📊 Test Results:")
    print(f"ML Model: {'✅ PASS' if ml_success else '❌ FAIL'}")
    print(f"API Endpoints: {'✅ PASS' if api_success else '❌ FAIL'}")
    
    if ml_success and api_success:
        print("\n🎉 All tests passed! The system is working correctly.")
        print("\n🌐 Access the application:")
        print("Frontend: http://localhost:3000")
        print("Backend API: http://localhost:8000")
        print("API Docs: http://localhost:8000/docs")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
    
    return ml_success and api_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

