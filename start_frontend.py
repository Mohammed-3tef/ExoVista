#!/usr/bin/env python3
"""
Simple frontend startup script for Cosmic Hunter
"""

import subprocess
import sys
import os

if __name__ == "__main__":
    print("🌐 Starting Cosmic Hunter Frontend...")
    print("Frontend will be available at: http://localhost:3000")
    print("Press Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        # Change to client directory
        os.chdir('client')
        
        # Start the HTTP server
        subprocess.run([
            sys.executable, '-m', 'http.server', '3000', 
            '--directory', 'public'
        ])
    except KeyboardInterrupt:
        print("\n🛑 Frontend server stopped")
    except Exception as e:
        print(f"❌ Frontend error: {e}")

