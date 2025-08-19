#!/usr/bin/env python3
"""
Startup script for the Adobe 1B Interface FastAPI server.
This replaces the old Express.js server completely.
"""

import uvicorn
from backend.app import app

if __name__ == "__main__":
    print("Starting Adobe 1B Interface FastAPI server...")
    print("Open your browser to: http://localhost:8080/")
    print("Press Ctrl+C to stop the server")
    
    uvicorn.run(
        "backend.app:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info"
    )
