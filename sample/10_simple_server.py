from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import torch
import numpy as np
import logging
from speechbrain.inference.TTS import FastSpeech2 
from speechbrain.inference.vocoders import HIFIGAN

# Setup logging to see errors in the console
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Streaming Multilingual TTS")

# ------------------ Load models ------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

mms_tts = FastSpeech2.from_hparams(
    source="speechbrain/tts-fastspeech2-mms",
    savedir="pretrained_mms",
    run_opts={"device": device}
)

hifigan = HIFIGAN.from_hparams(
    source="speechbrain/tts-hifigan-libritts-22050Hz",
    savedir="pretrained_hifigan",
    run_opts={"device": device}
)

@app.websocket("/ws_tts")
async def websocket_tts(ws: WebSocket):
    await ws.accept()
    try:
        text = await ws.receive_text()
        logger.info(f"Processing text: {text}")

        with torch.no_grad():
            # 1. Standardize text encoding
            # Some SpeechBrain versions require the text to be wrapped in a list for batching
            # and may need the specific 'encode_text' or 'generate_mel' method.
            
            # TRY THIS: Wrap text in a list to provide a batch dimension
            outputs = mms_tts.encode_text([text]) 
            
            # 2. Handle the output tuple
            # FastSpeech2 returns (mel_postnet, postnet_loss, durations, etc.)
            # We need the FIRST element which is the mel-spectrogram
            if isinstance(outputs, (tuple, list)):
                mel_output = outputs[0]
            else:
                mel_output = outputs

            # 3. Decode to Audio
            # Ensure the mel_output is on the correct device
            audio_tensor = hifigan.decode_batch(mel_output.to(device))
            
        # 4. Prepare for streaming
        audio = audio_tensor.squeeze().cpu().numpy()

        # Check if audio was actually generated
        if audio.size == 0:
            logger.error("Generated audio is empty")
            return

        chunk_size = 2048
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i : i + chunk_size]
            await ws.send_bytes(chunk.astype(np.float32).tobytes())

        logger.info("Streaming complete")
        
    except Exception as e:
        # This will now catch the exact line causing the "list index" error
        logger.error(f"Error during TTS processing: {str(e)}", exc_info=True)
    finally:
        try:
            await ws.close()
        except:
            pass