
For 1,2,3
(svastikkka) manshusharma@Manshus-MacBook-Air TTS % curl -X POST http://localhost:8000/tts \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello this is a from scratch TTS system","language":"en"}' \
  --output speech.wav

For 4,5
(svastikkka) manshusharma@Manshus-MacBook-Air TTS % curl -X POST http://localhost:8000/tts \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello this is a from scratch TTS system","language":"en"}'


For 6
curl -X POST http://localhost:8000/tts_audio \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello this is a from scratch TTS system","language":"en"}' \
  --output speech.wav



For 7,8
(svastikkka) manshusharma@Manshus-MacBook-Air sample % uvicorn 7_simple_server:app --host 0.0.0.0 --port 8000
(svastikkka) manshusharma@Manshus-MacBook-Air sample % python ./7_simple_client.py

(svastikkka) manshusharma@Manshus-MacBook-Air sample % uvicorn 8_simple_server:app --host 0.0.0.0 --port 8000
(svastikkka) manshusharma@Manshus-MacBook-Air sample % python ./8_simple_client.py

For 9
requirments.txt
```
# PyTorch CPU version (no GPU)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Or GPU version if you have CUDA
# pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# HiFi-GAN repo utilities
pip install git+https://github.com/jik876/hifi-gan.git

# Web server and audio utils
pip install fastapi uvicorn websockets soundfile numpy
pip install torch torchaudio fastapi uvicorn websockets soundfile numpy speechbrain==1.0.3

```

```
wget https://github.com/jik876/hifi-gan/raw/master/config_v2.json
```


For 10
```
pip install "huggingface-hub>=0.34.0" "transformers>=4.30.0" --upgrade
pip install torch torchaudio fastapi uvicorn websockets soundfile numpy speechbrain==1.0.3
```


For 11 and 12
```
pip install torch torchaudio
pip install numpy soundfile fastapi uvicorn websockets
pip install phonemizer
pip install unidic-lite
pip install pykakasi
python -m unidic download
pip install git+https://github.com/myshell-ai/MeloTTS.git

```