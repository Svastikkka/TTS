from melo.api import TTS
import soundfile as sf
import torch
import librosa

# 1. Setup Device
# ------------------ Device ------------------
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# 2. Initialize TTS for Indian English
# Use 'EN_V2' or just 'EN' depending on your installation
model = TTS(language='EN', device=device) 
speaker_ids = model.hps.data.spk2id

# 3. Choose the Indian Speaker
# In MeloTTS, this is usually 'EN_INDIA'
indian_speaker_id = speaker_ids['EN_INDIA']

# 4. Generate Hinglish-style Audio
# Since there is no native Devanagari (Hindi script) support, 
# write your Hindi words using Latin (Hinglish) characters.
text = "Namasteeee! Mera naam Gemini hai. I can speak a mix of Hindi and English fluently."
temp_output = "indian_accent_test.wav"

model.tts_to_file(text, indian_speaker_id, temp_output, speed=1.0)

# 5. Save and Load
sampling_rate = model.hps.data.sampling_rate
audio_np, _ = librosa.load(temp_output, sr=sampling_rate)
sf.write("melo_hindi_english.wav", audio_np, sampling_rate)

print(f"Success! Generated audio with {indian_speaker_id} accent.")