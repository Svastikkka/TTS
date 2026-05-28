import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Load audio
audio, sr = librosa.load(
    librosa.ex('trumpet'),
    sr=22050
)

print("Audio Shape:", audio.shape)
print("Sample Rate:", sr)

# Create mel spectrogram
mel = librosa.feature.melspectrogram(
    y=audio,
    sr=sr,
    n_fft=1024,
    hop_length=256,
    n_mels=80
)

print("Mel Shape:", mel.shape)

# Convert to dB
mel_db = librosa.power_to_db(mel, ref=np.max)

# Plot
plt.figure(figsize=(10, 4))

librosa.display.specshow(
    mel_db,
    sr=sr,
    hop_length=256,
    x_axis='time',
    y_axis='mel'
)

plt.colorbar(format='%+2.0f dB')
plt.title("Mel Spectrogram")
plt.tight_layout()
plt.show()