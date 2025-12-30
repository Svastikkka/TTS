"""
Docstring for sample.2_simple_server
Generating SIN Wave
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
import numpy as np
import soundfile as sf
import io

app = FastAPI(title="Simple TTS Server")

SAMPLE_RATE = 22050

class TTSRequest(BaseModel):
    text: str
    language: str = "en"

@app.post("/tts")
def tts(req: TTSRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    # 🔊 1-second sine wave (A4 = 440 Hz)
    t = np.linspace(0, 1, SAMPLE_RATE, endpoint=False)
    audio = 0.2 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    buffer = io.BytesIO()
    sf.write(buffer, audio, SAMPLE_RATE, format="WAV")
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="audio/wav",
        headers={
            "X-Sample-Rate": str(SAMPLE_RATE),
            "X-Language": req.language
        }
    )
