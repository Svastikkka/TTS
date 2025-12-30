import asyncio
import websockets
import numpy as np
import soundfile as sf

async def main():
    uri = "ws://localhost:8000/ws_tts"
    async with websockets.connect(uri) as ws:
        await ws.send("Hello this is a from scratch streaming TTS system")

        audio_chunks = []
        try:
            while True:
                data = await ws.recv()
                audio_frame = np.frombuffer(data, dtype=np.float32)
                audio_chunks.append(audio_frame)
        except websockets.ConnectionClosedOK:
            pass

        waveform = np.concatenate(audio_chunks)
        sf.write("speech_stream.wav", waveform, 22050)

asyncio.run(main())
