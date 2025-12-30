"""
Streaming TTS with Tacotron2 + HiFi-GAN Vocoder
Text → Tacotron2 → Mel → HiFi-GAN → Waveform chunks (WebSocket)
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import torch
import numpy as np
from speechbrain.pretrained import Tacotron2, HIFIGAN

app = FastAPI(title="Streaming TTS Server")

# ------------------ Load pretrained models ------------------
# Tacotron2: text -> mel
tacotron2 = Tacotron2.from_hparams(
    source="speechbrain/tts-tacotron2-ljspeech",
    savedir="pretrained_tts"
)

# HiFi-GAN: mel -> waveform
hifigan = HIFIGAN.from_hparams(
    source="speechbrain/tts-hifigan-libritts-22050Hz",
    savedir="pretrained_hifigan"
)

# ------------------ WebSocket endpoint ------------------
@app.websocket("/ws_tts")
async def websocket_tts(ws: WebSocket):
    await ws.accept()
    try:
        text = await ws.receive_text()

        # 1️⃣ Generate mel from Tacotron2
        with torch.no_grad():
            mel_output, mel_length, _ = tacotron2.encode_text(text)  # [1, n_mels, T]

        # 2️⃣ Decode waveform with HiFi-GAN
        with torch.no_grad():
            audio_tensor = hifigan.decode_batch(mel_output)  # [1, 1, T_audio]
        audio = audio_tensor.squeeze().cpu().numpy()

        # 3️⃣ Stream audio in chunks
        chunk_size = 2048
        idx = 0
        while idx < len(audio):
            chunk = audio[idx: idx+chunk_size]
            await ws.send_bytes(chunk.astype(np.float32).tobytes())
            idx += chunk_size

        await ws.close()

    except WebSocketDisconnect:
        print("Client disconnected")
