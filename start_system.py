#!/usr/bin/env python3
"""
Cosmic Hunter System Startup Script
Starts both backend and frontend servers
"""

import subprocess
import sys
import os
import time
import threading
import webbrowser
from pathlib import Path

def start_backend():
    """Start the FastAPI backend server"""
    print("🚀 Starting Cosmic Hunter Backend...")
    os.chdir('server')
    try:
        subprocess.run([sys.executable, 'main.py'], check=True, shell=True)
    except KeyboardInterrupt:
        print("\n🛑 Backend server stopped")
    except Exception as e:
        print(f"❌ Backend error: {e}")

def start_frontend():
    """Start the frontend server"""
    print("🌐 Starting Cosmic Hunter Frontend...")
    os.chdir('client')
    try:
        subprocess.run([sys.executable, '-m', 'http.server', '3000', '--directory', 'public'], check=True, shell=True)
    except KeyboardInterrupt:
        print("\n🛑 Frontend server stopped")
    except Exception as e:
        print(f"❌ Frontend error: {e}")

def check_dependencies():
    """Check if required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'fastapi', 'uvicorn', 'pandas', 'numpy', 
        'scikit-learn', 'xgboost', 'google-generativeai'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Please install them with: pip install " + " ".join(missing_packages))
        return False
    
    print("✅ All dependencies are installed")
    return True

def check_model_files():
    """Check if ML model files exist"""
    print("🔍 Checking ML model files...")
    
    model_files = [
        'ml/exoplanet_model.pkl',
        'ml/scaler.pkl',
        'ml/feature_importance.csv'
    ]
    
    missing_files = []
    
    for file_path in model_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing model files: {', '.join(missing_files)}")
        print("Please run the model training first: cd ml && python train_model.py")
        return False
    
    print("✅ All model files are present")
    return True

def main():
    """Main startup function"""
    print("🌟 Cosmic Hunter System Startup")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path('ml').exists() or not Path('server').exists() or not Path('client').exists():
        print("❌ Please run this script from the project root directory")
        sys.exit(1)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check model files
    if not check_model_files():
        sys.exit(1)
    
    print("\n🚀 Starting servers...")
    print("Backend will be available at: http://localhost:8000")
    print("Frontend will be available at: http://localhost:3000")
    print("API Documentation: http://localhost:8000/docs")
    print("\nPress Ctrl+C to stop all servers")
    print("-" * 50)
    
    # Start backend in a separate thread
    backend_thread = threading.Thread(target=start_backend, daemon=True)
    backend_thread.start()
    
    # Wait a moment for backend to start
    time.sleep(3)
    
    # Start frontend in a separate thread
    frontend_thread = threading.Thread(target=start_frontend, daemon=True)
    frontend_thread.start()
    
    # Wait a moment for frontend to start
    time.sleep(2)
    
    # Open browser
    try:
        webbrowser.open('http://localhost:3000')
        print("🌐 Opening browser...")
    except:
        print("⚠️ Could not open browser automatically")
    
    try:
        # Keep main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down Cosmic Hunter...")
        print("✅ System stopped successfully")

if __name__ == "__main__":
    main()
