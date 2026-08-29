"""
API Server Launcher Script
==========================
Run the local FastAPI server using Uvicorn on port 8000.
"""

import uvicorn

if __name__ == "__main__":
    print("=" * 80)
    print("Starting Unified Phishing Detection Local API Server...")
    print("Swagger Documentation: http://127.0.0.1:8000/docs")
    print("Health Check:          http://127.0.0.1:8000/health")
    print("=" * 80)
    uvicorn.run("api_server.main:app", host="127.0.0.1", port=8000, reload=False, log_level="info")
