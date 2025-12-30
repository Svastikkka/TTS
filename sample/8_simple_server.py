"""
From-scratch Low-Latency TTS – Streaming
Text → Phonemes → Acoustic Model → Mel → Waveform (chunked)

This version uses a formant-based DSP vocoder to produce speech-like audio.
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import numpy as np

app = FastAPI(title="From-Scratch Streaming TTS")

N_MELS = 80
FRAMES_PER_PHONEME = 5
PHONEME_EMBED_DIM = 64
SAMPLE_RATE = 22050
SAMPLES_PER_FRAME = 512  # larger frame for smoother audio

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

# Simple phoneme formants (frequency in Hz)
PHONEME_FORMANTS = {
    "AH": [700, 1100, 2450],
    "EH": [500, 1700, 2500],
    "IH": [400, 2000, 2550],
    "OW": [300, 870, 2240],
    "UH": [350, 600, 2700],
    "AA": [800, 1150, 2900],
    "SP": "silence",
    "S": "noise",
    "F": "noise",
    "HH": "noise",
    # Default for other consonants
}

# ------------------ Acoustic Model (fake) ------------------
class AcousticModel:
    def __init__(self):
        self.phoneme_embeddings = np.random.randn(NUM_PHONEMES, PHONEME_EMBED_DIM).astype(np.float32)
        self.projection = np.random.randn(PHONEME_EMBED_DIM, N_MELS).astype(np.float32)

    def infer_frame(self, pid):
        emb = self.phoneme_embeddings[pid]
        mel_frame = emb @ self.projection
        mel_frame = mel_frame / (np.max(np.abs(mel_frame)) + 1e-6)
        return mel_frame

# ------------------ Formant-based Vocoder ------------------
class Vocoder:
    def synthesize_frame(self, mel_frame, phoneme):
        frame_samples = np.arange(SAMPLES_PER_FRAME)
        sine_wave = np.zeros(SAMPLES_PER_FRAME, dtype=np.float32)

        if phoneme in PHONEME_FORMANTS:
            fdata = PHONEME_FORMANTS[phoneme]
            if fdata == "noise":
                sine_wave = np.random.randn(SAMPLES_PER_FRAME) * 0.1
            elif fdata == "silence":
                sine_wave = np.zeros(SAMPLES_PER_FRAME)
            else:
                for f in fdata:
                    sine_wave += np.sin(2 * np.pi * f * frame_samples / SAMPLE_RATE)
                sine_wave /= len(fdata)
        else:
            # fallback: use first mel value as frequency
            freq = 220 + 500 * np.tanh(mel_frame[0])
            sine_wave = np.sin(2 * np.pi * freq * frame_samples / SAMPLE_RATE)

        # Add slight noise for consonants
        sine_wave += np.random.randn(SAMPLES_PER_FRAME) * 0.01

        # Envelope
        envelope = np.linspace(1, 0.8, SAMPLES_PER_FRAME)
        sine_wave *= envelope

        # Normalize
        sine_wave /= np.max(np.abs(sine_wave) + 1e-6)
        return sine_wave.astype(np.float32)

acoustic_model = AcousticModel()
vocoder = Vocoder()

# ------------------ Phonemizer ------------------
def phonemize(text):
    phoneme_ids = []
    phonemes = []
    for ch in text.lower():
        ph = PHONEME_MAP.get(ch, "SP")
        phonemes.append(ph)
        phoneme_ids.append(PHONEME_ID_MAP[ph])
    return phonemes, phoneme_ids

# ------------------ WebSocket Endpoint ------------------
@app.websocket("/ws_tts")
async def websocket_tts(ws: WebSocket):
    await ws.accept()
    try:
        data = await ws.receive_text()
        phonemes, phoneme_ids = phonemize(data)

        for pid, ph in zip(phoneme_ids, phonemes):
            mel_frame = acoustic_model.infer_frame(pid)
            audio_frame = vocoder.synthesize_frame(mel_frame, ph)
            await ws.send_bytes(audio_frame.tobytes())

        # Close connection after sending all frames
        await ws.close()

    except WebSocketDisconnect:
        print("Client disconnected")
