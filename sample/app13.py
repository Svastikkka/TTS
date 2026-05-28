import os
import tarfile
import requests
import numpy as np
import pandas as pd
import librosa
import torch

from tqdm import tqdm
from torch.utils.data import (
    Dataset,
    DataLoader
)

# --------------------------------------------------
# DATASET CONFIG
# --------------------------------------------------

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

# --------------------------------------------------
# DOWNLOAD DATASET
# --------------------------------------------------

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

    print("Dataset ready!")

# --------------------------------------------------
# TOKENIZER
# --------------------------------------------------

characters = "abcdefghijklmnopqrstuvwxyz!'?,. "

char_to_id = {
    ch: i + 1
    for i, ch in enumerate(characters)
}

PAD_ID = 0

# --------------------------------------------------
# DATASET
# --------------------------------------------------

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

        # -------------------------
        # LOAD AUDIO
        # -------------------------

        audio, sr = librosa.load(
            wav_path,
            sr=22050
        )

        # -------------------------
        # MEL SPECTROGRAM
        # -------------------------

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

        # -------------------------
        # TOKENIZE TEXT
        # -------------------------

        text_seq = self.text_to_sequence(text)

        return text_seq, mel

# --------------------------------------------------
# COLLATE FUNCTION
# --------------------------------------------------

def collate_fn(batch):

    texts, mels = zip(*batch)

    # ------------------------------------------
    # TEXT PADDING
    # ------------------------------------------

    text_lengths = [
        len(t)
        for t in texts
    ]

    max_text_len = max(text_lengths)

    padded_texts = []

    for t in texts:

        padded = torch.nn.functional.pad(
            t,
            (0, max_text_len - len(t)),
            value=PAD_ID
        )

        padded_texts.append(padded)

    padded_texts = torch.stack(padded_texts)

    # ------------------------------------------
    # MEL PADDING
    # ------------------------------------------

    mel_lengths = [
        m.shape[1]
        for m in mels
    ]

    max_mel_len = max(mel_lengths)

    padded_mels = []

    for mel in mels:

        pad_amount = max_mel_len - mel.shape[1]

        padded = torch.nn.functional.pad(
            mel,
            (0, pad_amount),
            value=0
        )

        padded_mels.append(padded)

    padded_mels = torch.stack(padded_mels)

    return (
        padded_texts,
        padded_mels,
        torch.tensor(text_lengths),
        torch.tensor(mel_lengths)
    )

# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":

    download_dataset()

    dataset = LJDataset(EXTRACTED_PATH)

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn
    )

    batch = next(iter(dataloader))

    texts, mels, text_lengths, mel_lengths = batch

    print("\nTexts Shape:")
    print(texts.shape)

    print("\nMels Shape:")
    print(mels.shape)

    print("\nText Lengths:")
    print(text_lengths)

    print("\nMel Lengths:")
    print(mel_lengths)
