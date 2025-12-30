"""
From-scratch TTS (CORRECT ARCHITECTURE STAGE)
Text → Phonemes → Mel Spectrogram (placeholder)
NO waveform generation
NO TTS / ML libraries
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np

# --------------------------------------------------
# App
# --------------------------------------------------
app = FastAPI(title="From-Scratch TTS Server")

# Acoustic feature config (industry standard)
N_MELS = 80
FRAMES_PER_PHONEME = 5

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

PHONEMES = sorted(set(PHONEME_MAP.values()))
PHONEME_ID_MAP = {p: i for i, p in enumerate(PHONEMES)}

def phonemize(text: str):
    phonemes = []
    phoneme_ids = []
    durations = []

    for ch in text.lower():
        ph = PHONEME_MAP.get(ch, "SP")
        phonemes.append(ph)
        phoneme_ids.append(PHONEME_ID_MAP[ph])
        durations.append(FRAMES_PER_PHONEME)

    return phonemes, phoneme_ids, durations

# --------------------------------------------------
# Mel-spectrogram placeholder
# --------------------------------------------------
def acoustic_placeholder(phoneme_ids, durations):
    """
    Placeholder for TensorRT acoustic model output
    Shape: [n_mels, T]
    """

    total_frames = sum(durations)

    # Deterministic fake mel (for testing pipeline)
    mel = np.zeros((N_MELS, total_frames), dtype=np.float32)

    frame = 0
    for pid, dur in zip(phoneme_ids, durations):
        mel[:, frame : frame + dur] = pid / 10.0
        frame += dur

    return mel

# --------------------------------------------------
# TTS Endpoint
# --------------------------------------------------
@app.post("/tts")
def tts(req: TTSRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    phonemes, phoneme_ids, durations = phonemize(req.text)
    mel = acoustic_placeholder(phoneme_ids, durations)

    return {
        "language": req.language,
        "phonemes": phonemes,
        "phoneme_ids": phoneme_ids,
        "durations": durations,
        "mel_shape": list(mel.shape),
        "engine_stage": "acoustic-placeholder"
    }
