import asyncio
import websockets
import numpy as np
import soundfile as sf
from websockets.exceptions import ConnectionClosedOK, ConnectionClosedError

async def main():
    uri = "ws://localhost:8000/ws_tts"
    
    try:
        async with websockets.connect(uri) as ws:
            text_to_speak = "Hello! This is a streaming test."
            await ws.send(text_to_speak)

            audio_chunks = []
            print("Receiving audio...")
            
            try:
                while True:
                    # Added a timeout so it doesn't wait forever if server dies
                    data = await asyncio.wait_for(ws.recv(), timeout=10.0)
                    audio_frame = np.frombuffer(data, dtype=np.float32)
                    audio_chunks.append(audio_frame)
            except (ConnectionClosedOK, ConnectionClosedError):
                print("Connection closed by server.")
            except asyncio.TimeoutError:
                print("Timeout: Server took too long to respond.")

            if audio_chunks:
                waveform = np.concatenate(audio_chunks)
                sf.write("speech_stream.wav", waveform, 22050)
                print(f"Success! Saved {len(waveform)} samples to speech_stream.wav")
            else:
                print("Error: No audio data was received.")
                
    except Exception as e:
        print(f"Failed to connect: {e}")

if __name__ == "__main__":
    asyncio.run(main())