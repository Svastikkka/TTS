from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
import numpy as np
import soundfile as sf
import io

app = FastAPI()

class TTSRequest(BaseModel):
    text: str
    language: str = "en"

@app.post("/tts")
def tts(req: TTSRequest):
    # Dummy audio
    audio = np.random.randn(22050).astype(np.float32)

    buffer = io.BytesIO()
    sf.write(buffer, audio, 22050, format="WAV")
    buffer.seek(0)

    return StreamingResponse(buffer, media_type="audio/wav")
