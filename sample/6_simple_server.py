"""
From-scratch TTS – Acoustic Model + Vocoder Stage (TensorRT-ready)
Text → Phonemes → Acoustic Model → Mel Spectrogram → Waveform
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import numpy as np
import soundfile as sf
import io

# --------------------------------------------------
# App
# --------------------------------------------------
app = FastAPI(title="From-Scratch TTS Server")

# --------------------------------------------------
# Acoustic / Vocoder Config
# --------------------------------------------------
N_MELS = 80
FRAMES_PER_PHONEME = 5
PHONEME_EMBED_DIM = 64
SAMPLE_RATE = 22050
SAMPLES_PER_FRAME = 256  # how many waveform samples per mel frame

# --------------------------------------------------
# Request Schema
# --------------------------------------------------
class TTSRequest(BaseModel):
    text: str
    language: str = "en"

# --------------------------------------------------
# Phonemizer
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
NUM_PHONEMES = len(PHONEMES)

def phonemize(text: str):
    phonemes = []
    phoneme_ids = []
    durations = []
    for ch in text.lower():
        ph = PHONEME_MAP.get(ch, "SP")
        phonemes.append(ph)
        phoneme_ids.append(PHONEME_ID_MAP[ph])
        durations.append(FRAMES_PER_PHONEME)
    return phonemes, np.array(phoneme_ids, dtype=np.int64), np.array(durations, dtype=np.int64)

# --------------------------------------------------
# Acoustic Model (Placeholder)
# --------------------------------------------------
class AcousticModel:
    def __init__(self):
        self.phoneme_embeddings = np.random.randn(NUM_PHONEMES, PHONEME_EMBED_DIM).astype(np.float32)
        self.projection = np.random.randn(PHONEME_EMBED_DIM, N_MELS).astype(np.float32)

    def infer(self, phoneme_ids: np.ndarray, durations: np.ndarray) -> np.ndarray:
        total_frames = int(durations.sum())
        mel = np.zeros((N_MELS, total_frames), dtype=np.float32)
        frame = 0
        for pid, dur in zip(phoneme_ids, durations):
            emb = self.phoneme_embeddings[pid]
            mel_frame = emb @ self.projection
            mel[:, frame : frame + dur] = mel_frame[:, None]
            frame += dur
        mel = mel / (np.max(np.abs(mel)) + 1e-6)
        return mel

acoustic_model = AcousticModel()

# --------------------------------------------------
# Vocoder (DSP placeholder)
# --------------------------------------------------
class Vocoder:
    """
    Converts mel spectrogram → waveform.
    Simple deterministic DSP-based placeholder.
    """
    def synthesize(self, mel: np.ndarray) -> np.ndarray:
        n_mels, T = mel.shape
        total_samples = T * SAMPLES_PER_FRAME
        waveform = np.zeros(total_samples, dtype=np.float32)

        for t in range(T):
            freq = 220 + 880 * np.tanh(mel[0, t])  # map first mel to freq
            frame_samples = np.arange(SAMPLES_PER_FRAME)
            sine_wave = np.sin(2 * np.pi * freq * frame_samples / SAMPLE_RATE)
            waveform[t*SAMPLES_PER_FRAME:(t+1)*SAMPLES_PER_FRAME] = sine_wave

        waveform /= np.max(np.abs(waveform) + 1e-6)
        return waveform.astype(np.float32)

vocoder = Vocoder()

# --------------------------------------------------
# TTS Endpoint (Waveform Output)
# --------------------------------------------------
@app.post("/tts_audio")
def tts_audio(req: TTSRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    # 1️⃣ Phonemize
    phonemes, phoneme_ids, durations = phonemize(req.text)

    # 2️⃣ Acoustic Model → Mel
    mel = acoustic_model.infer(phoneme_ids, durations)

    # 3️⃣ Vocoder → Waveform
    waveform = vocoder.synthesize(mel)

    # 4️⃣ Stream as WAV
    buffer = io.BytesIO()
    sf.write(buffer, waveform, SAMPLE_RATE, format="WAV")
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="audio/wav",
        headers={
            "X-Sample-Rate": str(SAMPLE_RATE),
            "X-Language": req.language,
            "X-Engine": "acoustic+vocoder-placeholder"
        }
    )
