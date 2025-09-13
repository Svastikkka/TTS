from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from src.core.pipeline import TTSPipeline
import io

app = FastAPI(title="TTS Engine", version="1.0")

# Initialize pipeline once at startup
tts_pipeline = TTSPipeline()

class TTSRequest(BaseModel):
    text: str
    language: str  # ISO 639-1 code (e.g., 'en', 'fr', 'hi')

@app.post("/tts")
async def generate_speech(payload: TTSRequest):
    try:
        # Run full pipeline -> returns raw audio (float32 numpy array)
        audio_waveform, sample_rate = tts_pipeline.run(
            text=payload.text,
            language=payload.language
        )

        # Convert to WAV in memory
        import soundfile as sf
        buffer = io.BytesIO()
        sf.write(buffer, audio_waveform, sample_rate, format="WAV")
        buffer.seek(0)

        return StreamingResponse(buffer, media_type="audio/wav")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
