import uvicorn
import os

if __name__ == "__main__":
    # Allow configuration through environment variables
    host = os.getenv("TTS_HOST", "0.0.0.0")
    port = int(os.getenv("TTS_PORT", 8000))
    reload = os.getenv("TTS_RELOAD", "false").lower() == "true"

    # Start FastAPI server (defined in api/server.py)
    uvicorn.run(
        "src.api.server:app", 
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )
