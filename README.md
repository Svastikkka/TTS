# TTS
Text to speech model

### Diagram
```
LLM → [Language Detection] → [Text Normalization] → [Phoneme Generation] → [Acoustic Model] → [Vocoder] → Audio
```


### Run
```bash
# Build
docker build -t tts-engine .

# Run with GPU (requires NVIDIA Container Toolkit)
docker run --gpus all -p 8000:8000 tts-engine
```
