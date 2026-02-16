# TTS
Text to speech model

### Diagram
```
LLM → [Language Detection] → [Text Normalization] → [Phoneme Generation] → [Acoustic Model] → [Vocoder] → Audio
```

### Environment setup
1. Ensure we installed python 3.10 `python3.10 --version`
2. Create a virtual environment `python3.10 -m venv myenv`
3. Activate the environment `source myenv/bin/activate`
4. Verify activation `python --version`


### Load Data & Train Model
- To load data we will follow following steps
    - We first dowload the data in `data` folder using below command
        - `wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2`
        - `tar -xvjf LJSpeech-1.1.tar.bz2`
    - Now we will start our traing
        - `python src/train/trainer.py`
        - `python src/train/data_loader.py`
        - `python src/train/train.py`


### Run
```bash
# Build
docker build -t svastikkka/tts .

# Run with GPU (requires NVIDIA Container Toolkit)
docker run --gpus all -p 8000:8000 svastikkka/tts
```

# Model Characterstics
- Multi lingual
- Excellent at mixing languages (Hindi + English Mix)
- Very Low latency
- Can we be integrated with hifigan optional if it improve performance
- Add more languages
- Zero-shot language support
- Good quality without audio data
- Perfect accent without native speaker data