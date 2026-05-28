import os
import tarfile
import requests
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.data import (
    Dataset,
    DataLoader
)

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
# DATASET CONFIG
# ==================================================

DATASET_URL = (
    "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
)

DATA_DIR = "data"

ARCHIVE_PATH = os.path.join(
    DATA_DIR,
    "LJSpeech-1.1.tar.bz2"
)

EXTRACTED_PATH = os.path.join(
    DATA_DIR,
    "LJSpeech-1.1"
)

# ==================================================
# DOWNLOAD DATASET
# ==================================================

def download_dataset():

    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(EXTRACTED_PATH):
        print("Dataset already exists.")
        return

    if not os.path.exists(ARCHIVE_PATH):

        print("Downloading dataset...")

        response = requests.get(
            DATASET_URL,
            stream=True
        )

        total_size = int(
            response.headers.get("content-length", 0)
        )

        with open(ARCHIVE_PATH, "wb") as file:

            with tqdm(
                total=total_size,
                unit="B",
                unit_scale=True
            ) as progress:

                for chunk in response.iter_content(1024):

                    file.write(chunk)
                    progress.update(len(chunk))

    print("Extracting dataset...")

    with tarfile.open(
        ARCHIVE_PATH,
        "r:bz2"
    ) as tar:

        tar.extractall(DATA_DIR)

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
# DATASET
# ==================================================

class LJDataset(Dataset):

    def __init__(self, root_dir):

        self.root_dir = root_dir

        metadata_path = os.path.join(
            root_dir,
            "metadata.csv"
        )

        self.metadata = pd.read_csv(
            metadata_path,
            sep="|",
            header=None
        )

        # SMALL SUBSET FOR FAST LEARNING
        self.metadata = self.metadata[:100]

    def __len__(self):
        return len(self.metadata)

    def text_to_sequence(self, text):

        text = text.lower()

        sequence = []

        for ch in text:

            if ch in char_to_id:
                sequence.append(char_to_id[ch])

        return torch.tensor(sequence)

    def __getitem__(self, idx):

        row = self.metadata.iloc[idx]

        file_id = row[0]
        text = row[2]

        wav_path = os.path.join(
            self.root_dir,
            "wavs",
            file_id + ".wav"
        )

        # --------------------------------------
        # LOAD AUDIO
        # --------------------------------------

        audio, sr = librosa.load(
            wav_path,
            sr=22050
        )

        # --------------------------------------
        # MEL SPECTROGRAM
        # --------------------------------------

        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_fft=1024,
            hop_length=256,
            n_mels=80
        )

        mel = librosa.power_to_db(
            mel,
            ref=np.max
        )

        mel = torch.tensor(mel).float()

        # [80, T] -> [T, 80]
        mel = mel.transpose(0, 1)

        text_seq = self.text_to_sequence(text)

        return text_seq, mel

# ==================================================
# COLLATE FUNCTION
# ==================================================

def collate_fn(batch):

    texts, mels = zip(*batch)

    # --------------------------------------
    # TEXT PADDING
    # --------------------------------------

    text_lengths = [len(t) for t in texts]
    max_text_len = max(text_lengths)

    padded_texts = []

    for t in texts:

        padded = F.pad(
            t,
            (0, max_text_len - len(t)),
            value=PAD_ID
        )

        padded_texts.append(padded)

    padded_texts = torch.stack(padded_texts)

    # --------------------------------------
    # MEL PADDING
    # --------------------------------------

    mel_lengths = [m.shape[0] for m in mels]
    max_mel_len = max(mel_lengths)

    padded_mels = []

    for mel in mels:

        pad_amount = max_mel_len - mel.shape[0]

        padded = F.pad(
            mel,
            (0, 0, 0, pad_amount)
        )

        padded_mels.append(padded)

    padded_mels = torch.stack(padded_mels)

    return (
        padded_texts,
        padded_mels,
        torch.tensor(text_lengths),
        torch.tensor(mel_lengths)
    )

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

        # ----------------------------------
        # TEMPORAL UPSAMPLING
        # ----------------------------------

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
# MAIN
# ==================================================

download_dataset()

dataset = LJDataset(EXTRACTED_PATH)

dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=collate_fn
)

model = TinyTTS().to(device)

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

criterion = nn.MSELoss()

# ==================================================
# TRAINING LOOP
# ==================================================

epochs = 5

for epoch in range(epochs):

    total_loss = 0

    for batch in dataloader:

        texts, mels, text_lens, mel_lens = batch

        texts = texts.to(device)
        mels = mels.to(device)

        max_mel_len = mels.shape[1]

        # ----------------------------------
        # FORWARD
        # ----------------------------------

        predicted_mels = model(
            texts,
            max_mel_len
        )

        # ----------------------------------
        # LOSS
        # ----------------------------------

        loss = criterion(
            predicted_mels,
            mels
        )

        # ----------------------------------
        # BACKPROP
        # ----------------------------------

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)

    print(
        f"Epoch {epoch+1} "
        f"Loss: {avg_loss:.4f}"
    )

    torch.save(
        model.state_dict(),
        "tiny_tts.pth"
    )
    print("Model Saved!")