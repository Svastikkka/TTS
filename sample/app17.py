import librosa
import torch
import soundfile as sf
import numpy as np

# ==================================================
# DEVICE
# ==================================================

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print("Using Device:", device)

# ==================================================
# LOAD HIFIGAN
# ==================================================

hifigan, _, _ = torch.hub.load(
    'NVIDIA/DeepLearningExamples:torchhub',
    'nvidia_hifigan',
    trust_repo=True
)

hifigan = hifigan.to(device)

hifigan.eval()

print("HiFiGAN Loaded!")

# ==================================================
# LOAD SAMPLE AUDIO
# ==================================================

audio, sr = librosa.load(
    librosa.ex('libri1'),
    sr=22050
)

print("Original Audio Shape:", audio.shape)

# ==================================================
# CREATE MEL SPECTROGRAM
# ==================================================

# IMPORTANT:
# These parameters should match HiFiGAN expectations

mel = librosa.feature.melspectrogram(
    y=audio,
    sr=22050,
    n_fft=1024,
    hop_length=256,
    win_length=1024,
    n_mels=80,
    fmin=0,
    fmax=8000
)

# Convert power mel -> log mel
mel = np.log(
    np.clip(
        mel,
        a_min=1e-5,
        a_max=None
    )
)

print("Mel Shape:", mel.shape)

# ==================================================
# PREPARE INPUT
# ==================================================

mel_tensor = torch.tensor(
    mel
).unsqueeze(0).float().to(device)

print("Mel Tensor Shape:", mel_tensor.shape)

# ==================================================
# GENERATE AUDIO
# ==================================================

with torch.no_grad():

    generated_audio = hifigan(
        mel_tensor
    )

# ==================================================
# CONVERT TO NUMPY
# ==================================================

generated_audio = (
    generated_audio
    .squeeze()
    .cpu()
    .numpy()
)

print("Generated Audio Shape:")
print(generated_audio.shape)

# ==================================================
# NORMALIZE AUDIO
# ==================================================

generated_audio = generated_audio / (
    np.max(np.abs(generated_audio)) + 1e-8
)

# ==================================================
# SAVE AUDIO
# ==================================================

sf.write(
    "generated.wav",
    generated_audio,
    22050
)

print("Saved: generated.wav")
