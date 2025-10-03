#!/usr/bin/env python3
"""
Complete System Test for Cosmic Hunter
Tests all components: ML model, backend API, and frontend integration
"""

import sys
import os
import requests
import time
import subprocess
import threading
import json
from pathlib import Path

class CosmicHunterTester:
    def __init__(self):
        self.api_base_url = "http://localhost:8000"
        self.frontend_url = "http://localhost:3000"
        self.backend_process = None
        self.frontend_process = None
        
    def test_ml_model(self):
        """Test the ML model directly"""
        print("🧪 Testing ML Model...")
        
        try:
            sys.path.append('ml')
            from ml.exoplanet_utils import ExoplanetPredictor, create_sample_data

            
            # Test model loading with correct paths
            model_path = os.path.join('ml', 'exoplanet_model.pkl')
            scaler_path = os.path.join('ml', 'scaler.pkl')
            predictor = ExoplanetPredictor(model_path, scaler_path)
            print("✅ Model loaded successfully")
            
            # Test prediction
            sample_data = create_sample_data()
            result = predictor.predict_single(sample_data)
            print(f"✅ Sample prediction: {result['classification']} (confidence: {result['confidence']:.3f})")
            
            return True
        except Exception as e:
            print(f"❌ ML Model test failed: {e}")
            return False
    
    def start_servers(self):
        """Start backend and frontend servers"""
        print("🚀 Starting servers...")
        
        # Start backend
        self.backend_process = subprocess.Popen(
            [sys.executable, 'main.py'],
            cwd='server',
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True
        )
        
        # Wait for backend to start
        time.sleep(5)
        
        # Start frontend
        self.frontend_process = subprocess.Popen(
            [sys.executable, '-m', 'http.server', '3000', '--directory', 'public'],
            cwd='client',
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True
        )
        
        # Wait for frontend to start
        time.sleep(3)
        
        print("✅ Servers started")
    
    def stop_servers(self):
        """Stop all servers"""
        if self.backend_process:
            self.backend_process.terminate()
        if self.frontend_process:
            self.frontend_process.terminate()
        print("🛑 Servers stopped")
    
    def test_backend_health(self):
        """Test backend health endpoint"""
        print("🌐 Testing backend health...")
        
        try:
            response = requests.get(f"{self.api_base_url}/health", timeout=10)
            if response.status_code == 200:
                print("✅ Backend health check passed")
                return True
            else:
                print(f"❌ Backend health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Backend health check error: {e}")
            return False
    
    def test_single_prediction(self):
        """Test single exoplanet prediction"""
        print("🔮 Testing single prediction...")
        
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
        
        try:
            response = requests.post(
                f"{self.api_base_url}/api/predict/single",
                json=sample_data,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Single prediction: {result['classification']} ({result['confidence']:.3f})")
                return True
            else:
                print(f"❌ Single prediction failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Single prediction error: {e}")
            return False
    
    def test_batch_prediction(self):
        """Test batch prediction with sample CSV"""
        print("📊 Testing batch prediction...")
        
        # Create sample CSV data
        csv_content = """koi_ror,koi_impact,koi_depth,koi_prad,koi_teq,koi_duration,koi_insol,koi_steff
0.02,0.3,400.0,1.5,1200.0,3.5,1000.0,5500.0
0.015,0.2,300.0,1.2,1100.0,3.0,900.0,5200.0"""
        
        try:
            files = {'file': ('test_data.csv', csv_content, 'text/csv')}
            response = requests.post(
                f"{self.api_base_url}/api/predict/batch",
                files=files,
                timeout=15
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Batch prediction: {result['successful_predictions']}/{result['total_processed']} successful")
                return True
            else:
                print(f"❌ Batch prediction failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Batch prediction error: {e}")
            return False
    
    def test_chat_endpoint(self):
        """Test chat endpoint"""
        print("💬 Testing chat endpoint...")
        
        try:
            response = requests.post(
                f"{self.api_base_url}/api/chat",
                json={"message": "What is an exoplanet?"},
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Chat response received: {len(result['reply'])} characters")
                return True
            else:
                print(f"❌ Chat endpoint failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Chat endpoint error: {e}")
            return False
    
    def test_model_info(self):
        """Test model info endpoint"""
        print("ℹ️ Testing model info...")
        
        try:
            response = requests.get(f"{self.api_base_url}/api/model/info", timeout=10)
            if response.status_code == 200:
                info = response.json()
                print(f"✅ Model info: {info['model_type']} with {info['feature_count']} features")
                return True
            else:
                print(f"❌ Model info failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Model info error: {e}")
            return False
    
    def test_frontend_accessibility(self):
        """Test if frontend is accessible"""
        print("🌐 Testing frontend accessibility...")
        
        try:
            response = requests.get(self.frontend_url, timeout=10)
            if response.status_code == 200:
                print("✅ Frontend is accessible")
                return True
            else:
                print(f"❌ Frontend not accessible: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Frontend accessibility error: {e}")
            return False
    
    def run_all_tests(self):
        """Run all tests"""
        print("🌟 Cosmic Hunter Complete System Test")
        print("=" * 60)
        
        results = {}
        
        # Test ML model
        results['ml_model'] = self.test_ml_model()
        
        # Start servers
        self.start_servers()
        
        # Test backend endpoints
        results['backend_health'] = self.test_backend_health()
        results['single_prediction'] = self.test_single_prediction()
        results['batch_prediction'] = self.test_batch_prediction()
        results['chat_endpoint'] = self.test_chat_endpoint()
        results['model_info'] = self.test_model_info()
        
        # Test frontend
        results['frontend_access'] = self.test_frontend_accessibility()
        
        # Stop servers
        self.stop_servers()
        
        # Print results
        print("\n📊 Test Results Summary:")
        print("-" * 40)
        
        total_tests = len(results)
        passed_tests = sum(results.values())
        
        for test_name, passed in results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{test_name.replace('_', ' ').title()}: {status}")
        
        print("-" * 40)
        print(f"Overall: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            print("\n🎉 All tests passed! The system is working perfectly!")
            print("\n🌐 Access the application:")
            print("Frontend: http://localhost:3000")
            print("Backend API: http://localhost:8000")
            print("API Docs: http://localhost:8000/docs")
        else:
            print(f"\n❌ {total_tests - passed_tests} tests failed. Please check the errors above.")
        
        return passed_tests == total_tests

def main():
    """Main test function"""
    tester = CosmicHunterTester()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
