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
# LOAD SAMPLE AUDIO
# ==================================================

audio, sr = librosa.load(
    librosa.ex('libri1'),
    sr=22050
)

print("Audio Shape:", audio.shape)

# ==================================================
# CREATE MEL
# ==================================================

mel = librosa.feature.melspectrogram(
    y=audio,
    sr=sr,
    n_fft=1024,
    hop_length=256,
    n_mels=80
)

mel = np.log(
    np.clip(mel, a_min=1e-5, a_max=None)
)

print("Mel Shape:", mel.shape)

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

generated_audio = (
    generated_audio
    .squeeze()
    .cpu()
    .numpy()
)

print("Generated Audio Shape:")
print(generated_audio.shape)

# ==================================================
# SAVE
# ==================================================

sf.write(
    "generated.wav",
    generated_audio,
    22050
)

print("Saved: generated.wav")
