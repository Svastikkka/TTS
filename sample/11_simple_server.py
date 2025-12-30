from melo.api import TTS
import soundfile as sf
import torch
import os
import librosa

# 1. Setup Device (MPS for Mac M-series speed)
device = "mps" if torch.backends.mps.is_available() else "cpu"

# 2. Initialize TTS
# Using 'EN' for English. For Hindi-English, we will discuss the checkpoint next.
model = TTS(language='EN', device=device)
speaker_ids = model.hps.data.spk2id

# 3. Generate Audio
text = "Hello! The dictionary issue is fixed, and Melo TTS is now running on my Mac."
temp_output = "temp_output.wav"

# MeloTTS uses 'tts_to_file' as its main interface.
# We use 'EN-Default' for a clean neutral accent.
model.tts_to_file(text, speaker_ids['EN-Default'], temp_output, speed=1.0)

# 4. Correctly access the sample rate
# The attribute is .hps (HyperParameters) not .hparams
sampling_rate = model.hps.data.sampling_rate

# 5. Load it back as a numpy array using the detected rate
audio_np, _ = librosa.load(temp_output, sr=sampling_rate)

# 6. Save final version
sf.write("melo_test_fixed.wav", audio_np, sampling_rate)

print(f"Success! Audio saved at {sampling_rate}Hz using speaker {speaker_ids['EN-Default']}")