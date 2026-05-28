import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np
import soundfile as sf

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
# TOKENIZER
# ==================================================

characters = "abcdefghijklmnopqrstuvwxyz!'?,. "

char_to_id = {
    ch: i + 1
    for i, ch in enumerate(characters)
}

PAD_ID = 0
VOCAB_SIZE = len(char_to_id) + 1

# ==================================================
# MODEL
# ==================================================

class TinyTTS(nn.Module):

    def __init__(self):

        super().__init__()

        self.embedding = nn.Embedding(
            VOCAB_SIZE,
            128,
            padding_idx=PAD_ID
        )

        self.encoder = nn.LSTM(
            input_size=128,
            hidden_size=256,
            batch_first=True,
            bidirectional=True
        )

        self.projection = nn.Linear(
            512,
            80
        )

    def forward(
        self,
        text,
        mel_len
    ):

        x = self.embedding(text)

        x, _ = self.encoder(x)

        # --------------------------------------
        # TEMPORAL UPSAMPLING
        # --------------------------------------

        x = x.transpose(1, 2)

        x = F.interpolate(
            x,
            size=mel_len,
            mode="linear"
        )

        x = x.transpose(1, 2)

        mel = self.projection(x)

        return mel

# ==================================================
# LOAD MODEL
# ==================================================

model = TinyTTS().to(device)

model.load_state_dict(
    torch.load(
        "tiny_tts.pth",
        map_location=device
    )
)

model.eval()

print("Model Loaded!")

# ==================================================
# TOKENIZER
# ==================================================

def text_to_sequence(text):

    text = text.lower()

    sequence = []

    for ch in text:

        if ch in char_to_id:
            sequence.append(char_to_id[ch])

    return torch.tensor(sequence)

# ==================================================
# INPUT TEXT
# ==================================================

text = "hello world"

tokens = text_to_sequence(text)

tokens = tokens.unsqueeze(0).to(device)

print("Token Shape:", tokens.shape)

# ==================================================
# GENERATE MEL
# ==================================================

# Fake mel length estimate
mel_len = tokens.shape[1] * 8

with torch.no_grad():

    predicted_mel = model(
        tokens,
        mel_len
    )

print("Predicted Mel Shape:")
print(predicted_mel.shape)

# ==================================================
# MEL -> AUDIO
# ==================================================

# [1, T, 80]
mel = predicted_mel[0]

# [T, 80] -> [80, T]
mel = mel.transpose(0, 1)

mel = mel.cpu().numpy()

# Convert dB -> power
mel_power = librosa.db_to_power(mel)

# Griffin-Lim vocoder
audio = librosa.feature.inverse.mel_to_audio(
    mel_power,
    sr=22050,
    n_fft=1024,
    hop_length=256
)

# ==================================================
# SAVE AUDIO
# ==================================================

sf.write(
    "generated.wav",
    audio,
    22050
)

print("Generated speech saved!")
