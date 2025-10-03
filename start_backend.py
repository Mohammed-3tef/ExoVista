#!/usr/bin/env python3
"""
Simple backend startup script for Cosmic Hunter
"""

import uvicorn
import os
import sys

# Add the current directory to Python path
sys.path.append(os.path.dirname(__file__))

if __name__ == "__main__":
    print("🚀 Starting Cosmic Hunter Backend...")
    print("Backend will be available at: http://localhost:8000")
    print("API Documentation: http://localhost:8000/docs")
    print("Press Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=False,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n🛑 Backend server stopped")
    except Exception as e:
        print(f"❌ Backend error: {e}")

