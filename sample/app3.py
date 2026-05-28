import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

# Load sample audio
audio, sr = librosa.load(
    librosa.ex('libri1'),
    sr=22050
)

print("Original Audio Shape:", audio.shape)

# Create mel spectrogram
mel = librosa.feature.melspectrogram(
    y=audio,
    sr=sr,
    n_fft=1024,
    hop_length=256,
    n_mels=80
)

print("Mel Shape:", mel.shape)

# Convert mel → audio using Griffin-Lim
reconstructed_audio = librosa.feature.inverse.mel_to_audio(
    mel,
    sr=sr,
    n_fft=1024,
    hop_length=256
)

print("Reconstructed Audio Shape:", reconstructed_audio.shape)

# Save audio
sf.write(
    "output/output.wav",
    reconstructed_audio,
    sr
)

print("Saved: output/output.wav")