import os
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
import soundfile as sf

# ==================================================
# DEVICE (Explicitly targeting cuda:1 to match training)
# ==================================================

if torch.cuda.is_available():
    device = "cuda:1" 
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print("Using Device:", device)

# ==================================================
# CONFIG
# ==================================================

N_MELS = 80
PAD_ID = 0

characters = "abcdefghijklmnopqrstuvwxyz!'?,. "
char_to_id = {ch: i + 1 for i, ch in enumerate(characters)}
VOCAB_SIZE = len(char_to_id) + 1

# ==================================================
# TOKENIZER
# ==================================================

def text_to_sequence(text):
    text = text.lower().strip()
    sequence = []
    for ch in text:
        if ch in char_to_id:
            sequence.append(char_to_id[ch])
    return torch.tensor(sequence, dtype=torch.long)

# ==================================================
# LENGTH REGULATOR
# ==================================================

class LengthRegulator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, encoder_output, durations):
        expanded_batch = []
        batch_size = encoder_output.shape[0]

        for b in range(batch_size):
            expanded = []
            for i, duration in enumerate(durations[b]):
                duration = max(1, int(duration.item()))
                repeated = encoder_output[b, i].unsqueeze(0).repeat(duration, 1)
                expanded.append(repeated)

            expanded = torch.cat(expanded, dim=0)
            expanded_batch.append(expanded)

        max_len = max(x.shape[0] for x in expanded_batch)
        padded_batch = []
        for x in expanded_batch:
            pad_len = max_len - x.shape[0]
            padded = nn.functional.pad(x, (0, 0, 0, pad_len))
            padded_batch.append(padded)

        return torch.stack(padded_batch)

# ==================================================
# DURATION PREDICTOR
# ==================================================

class DurationPredictor(nn.Module):
    def __init__(self, input_dim=512): # Updated to receive dynamic dimension from larger encoder
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, 256, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.layer_norm1 = nn.LayerNorm(256)
        self.conv2 = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.layer_norm2 = nn.LayerNorm(256)
        self.linear = nn.Linear(256, 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.relu1(x)
        x = x.transpose(1, 2)
        x = self.layer_norm1(x)
        x = x.transpose(1, 2)
        x = self.conv2(x)
        x = self.relu2(x)
        x = x.transpose(1, 2)
        x = self.layer_norm2(x)
        durations = self.linear(x)
        durations = torch.relu(durations)
        durations = durations.squeeze(-1)
        return durations

# ==================================================
# FASTSPEECH MODEL
# ==================================================

class MiniFastSpeech(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, 256, padding_idx=PAD_ID) # Scaled up to 256
        self.encoder = nn.LSTM(
            input_size=256, # Scaled input
            hidden_size=256, # Scaled hidden states
            num_layers=2, # Increased depth to 2 layers
            batch_first=True, 
            bidirectional=True
        )
        self.duration_predictor = DurationPredictor(input_dim=512) # Receives 256 * 2 due to bidirectionality
        self.length_regulator = LengthRegulator()
        self.decoder = nn.LSTM(
            input_size=512, # Accepts encoder context output shape
            hidden_size=256, # Scaled hidden states
            num_layers=2, # Increased depth to 2 layers
            batch_first=True
        )
        self.mel_projection = nn.Linear(256, N_MELS)

    def forward(self, text, durations=None):
        x = self.embedding(text)
        x, _ = self.encoder(x)
        predicted_durations = self.duration_predictor(x)

        if durations is None:
            durations = torch.round(predicted_durations).long()
            durations = torch.clamp(durations, min=1)

        x = self.length_regulator(x, durations)
        x, _ = self.decoder(x)
        mel = self.mel_projection(x)
        return mel, predicted_durations

# ==================================================
# LOAD MODELS
# ==================================================

model = MiniFastSpeech().to(device)
model.load_state_dict(torch.load("mini_fastspeech.pth", map_location=device))
model.eval()
print("Model Loaded!")

hifigan, _, _ = torch.hub.load(
    'NVIDIA/DeepLearningExamples:torchhub',
    'nvidia_hifigan',
    trust_repo=True
)
hifigan = hifigan.to(device)
hifigan.eval()
print("HiFiGAN Loaded!")

# ==================================================
# INPUT TEXT
# ==================================================

text = "hello world"
tokens = text_to_sequence(text).unsqueeze(0).to(device)

# ==================================================
# GENERATE MEL
# ==================================================

with torch.no_grad():
    predicted_mel, _ = model(tokens)

mel = predicted_mel[0].transpose(0, 1)  # [T, 80] -> [80, T]

# --------------------------------------------------
# FIX: CONVERT NATURAL LOG MEL TO NVIDIA BASE-10 ENERGY SCALE
# --------------------------------------------------
# 1. Reverse the natural log transformation to return to linear power amplitudes
linear_mel = torch.exp(mel)

# 2. Re-scale to NVIDIA's precise Base-10 db format expected by this HiFi-GAN version
nvidia_mel = torch.log10(torch.clamp(linear_mel, min=1e-5)) * 1.0

# 3. Apply the specific HiFi-GAN amplitude scaling offset constant
nvidia_mel = (nvidia_mel * 3.2582)

# Prepare batch dimension for deep learning pipeline processing
nvidia_mel = nvidia_mel.unsqueeze(0)

# ==================================================
# GENERATE AUDIO
# ==================================================

with torch.no_grad():
    audio = hifigan(nvidia_mel)

audio = audio.squeeze().cpu().numpy()

# Normalize
if np.max(np.abs(audio)) > 0:
    audio = audio / (np.max(np.abs(audio)) + 1e-8)

sf.write("generated.wav", audio, 22050)
print("Saved: generated.wav")