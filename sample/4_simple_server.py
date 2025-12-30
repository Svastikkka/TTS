"""
From-scratch TTS placeholder
Text → Phonemes → DSP-based speech-shaped audio
NO TTS / ML libraries
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
import numpy as np
import soundfile as sf
import io

# --------------------------------------------------
# App
# --------------------------------------------------
app = FastAPI(title="From-Scratch TTS Server")

SAMPLE_RATE = 22050

# --------------------------------------------------
# Request Schema
# --------------------------------------------------
class TTSRequest(BaseModel):
    text: str
    language: str = "en"

# --------------------------------------------------
# Simple rule-based phonemizer (English only)
# --------------------------------------------------
PHONEME_MAP = {
    "a": "AH", "b": "B", "c": "K", "d": "D", "e": "EH",
    "f": "F", "g": "G", "h": "HH", "i": "IH", "j": "JH",
    "k": "K", "l": "L", "m": "M", "n": "N", "o": "OW",
    "p": "P", "q": "K", "r": "R", "s": "S", "t": "T",
    "u": "UH", "v": "V", "w": "W", "x": "KS", "y": "Y",
    "z": "Z",
    " ": "SP"
}

def phonemize(text: str):
    """
    Convert text to a phoneme sequence.
    VERY naive, but linguistically meaningful.
    """
    phonemes = []
    for ch in text.lower():
        phonemes.append(PHONEME_MAP.get(ch, "SP"))
    return phonemes

# --------------------------------------------------
# DSP Speech-like Synth driven by phonemes
# --------------------------------------------------
def synthesize_from_phonemes(phonemes, sr: int) -> np.ndarray:
    """
    Generate speech-shaped audio using phoneme timing.
    """

    phoneme_duration = 0.08  # seconds per phoneme
    total_duration = max(len(phonemes) * phoneme_duration, 0.5)

    signal = []

    for ph in phonemes:
        length = int(sr * phoneme_duration)
        t = np.linspace(0, phoneme_duration, length, endpoint=False)

        # Silence
        if ph == "SP":
            frame = np.zeros(length)
        else:
            # Map phoneme to base frequency (fake articulation)
            base_freq = 100 + (hash(ph) % 200)

            voiced = np.sin(2 * np.pi * base_freq * t)
            noise = np.random.randn(length) * 0.2

            frame = 0.8 * voiced + 0.2 * noise

            # Envelope
            envelope = np.exp(-5 * t / phoneme_duration)
            frame *= envelope

        signal.append(frame)

    signal = np.concatenate(signal)

    # Normalize
    signal /= np.max(np.abs(signal) + 1e-6)

    return signal.astype(np.float32)

# --------------------------------------------------
# TTS Endpoint
# --------------------------------------------------
@app.post("/tts")
def tts(req: TTSRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    phonemes = phonemize(req.text)
    audio = synthesize_from_phonemes(phonemes, SAMPLE_RATE)

    buffer = io.BytesIO()
    sf.write(buffer, audio, SAMPLE_RATE, format="WAV")
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="audio/wav",
        headers={
            "X-Sample-Rate": str(SAMPLE_RATE),
            "X-Language": req.language,
            "X-Engine": "from-scratch-phoneme-dsp",
            "X-Phoneme-Count": str(len(phonemes))
        }
    )
