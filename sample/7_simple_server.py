"""
From-scratch Low-Latency TTS – Streaming
Text → Phonemes → Acoustic Model → Mel → Waveform (chunked)

Streaming / Low-Latency TTS. This will allow:

- Real-time phoneme → mel → waveform generation
- Sending audio in chunks over WebSocket
- Latency of <200ms instead of generating the full sentence first
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import numpy as np

app = FastAPI(title="From-Scratch Streaming TTS")

N_MELS = 80
FRAMES_PER_PHONEME = 5
PHONEME_EMBED_DIM = 64
SAMPLE_RATE = 22050
SAMPLES_PER_FRAME = 256

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

# Reuse your acoustic model + vocoder
class AcousticModel:
    def __init__(self):
        self.phoneme_embeddings = np.random.randn(NUM_PHONEMES, PHONEME_EMBED_DIM).astype(np.float32)
        self.projection = np.random.randn(PHONEME_EMBED_DIM, N_MELS).astype(np.float32)
    def infer_frame(self, pid):
        emb = self.phoneme_embeddings[pid]
        mel_frame = emb @ self.projection
        mel_frame = mel_frame / (np.max(np.abs(mel_frame)) + 1e-6)
        return mel_frame

class Vocoder:
    def synthesize_frame(self, mel_frame):
        base_freq = 100 + 500 * np.tanh(mel_frame[0])  # base pitch

        frame_samples = np.arange(SAMPLES_PER_FRAME)
        sine_wave = np.zeros(SAMPLES_PER_FRAME)

        # Add 3 harmonics
        for i in range(1, 4):
            sine_wave += (0.5 / i) * np.sin(2 * np.pi * base_freq * i * frame_samples / SAMPLE_RATE)

        # Add slight noise (unvoiced sounds)
        sine_wave += np.random.randn(SAMPLES_PER_FRAME) * 0.05

        # Apply short envelope to smooth
        envelope = np.linspace(1, 0.7, SAMPLES_PER_FRAME)
        sine_wave *= envelope

        # Normalize
        sine_wave /= np.max(np.abs(sine_wave) + 1e-6)
        return sine_wave.astype(np.float32)

acoustic_model = AcousticModel()
vocoder = Vocoder()

def phonemize(text):
    phoneme_ids = [PHONEME_ID_MAP.get(PHONEME_MAP.get(ch, "SP"), 0) for ch in text.lower()]
    return phoneme_ids

@app.websocket("/ws_tts")
async def websocket_tts(ws: WebSocket):
    await ws.accept()
    try:
        data = await ws.receive_text()
        phoneme_ids = phonemize(data)

        for pid in phoneme_ids:
            mel_frame = acoustic_model.infer_frame(pid)
            audio_frame = vocoder.synthesize_frame(mel_frame)
            await ws.send_bytes(audio_frame.tobytes())

        # ✅ After sending all frames, close connection
        await ws.close()
    except WebSocketDisconnect:
        print("Client disconnected")

