"""
Docstring for sample.3_simple_server
Generating Musical Note
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
import numpy as np
import soundfile as sf
import io

# ------------------------------------------------------------------
# App Config
# ------------------------------------------------------------------
app = FastAPI(title="From-Scratch TTS Server")

SAMPLE_RATE = 22050

# ------------------------------------------------------------------
# Request Schema
# ------------------------------------------------------------------
class TTSRequest(BaseModel):
    text: str
    language: str = "en"

# ------------------------------------------------------------------
# DSP-based Speech-like Synth (Placeholder for ML)
# ------------------------------------------------------------------
def synthesize_speech_like(text: str, sample_rate: int) -> np.ndarray:
    """
    Speech-like waveform generator (NO TTS libs).
    This will later be replaced by TensorRT inference.
    """

    # Duration scales with text length
    duration = max(0.06 * len(text), 0.5)
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    # Base pitch changes slightly with text
    base_freq = 110 + (len(text) % 6) * 15

    signal = np.zeros_like(t)

    # Add harmonics (formant-like structure)
    for i in range(1, 6):
        signal += (1 / i) * np.sin(2 * np.pi * base_freq * i * t)

    # Simple amplitude envelope
    envelope = np.exp(-3 * t / duration)
    signal *= envelope

    # Normalize
    signal /= np.max(np.abs(signal) + 1e-6)

    return signal.astype(np.float32)

# ------------------------------------------------------------------
# TTS Endpoint
# ------------------------------------------------------------------
@app.post("/tts")
def tts(req: TTSRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    audio = synthesize_speech_like(req.text, SAMPLE_RATE)

    buffer = io.BytesIO()
    sf.write(buffer, audio, SAMPLE_RATE, format="WAV")
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="audio/wav",
        headers={
            "X-Sample-Rate": str(SAMPLE_RATE),
            "X-Language": req.language,
            "X-Engine": "from-scratch-dsp"
        }
    )
