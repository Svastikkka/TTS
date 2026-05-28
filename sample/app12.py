import os
import tarfile
import requests
import numpy as np
import pandas as pd
import librosa
import torch

from tqdm import tqdm
from torch.utils.data import Dataset

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

    # Skip if already extracted
    if os.path.exists(EXTRACTED_PATH):
        print("Dataset already exists.")
        return

    # Download archive
    if not os.path.exists(ARCHIVE_PATH):

        print("Downloading LJSpeech dataset...")

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

    # Extract
    print("Extracting dataset...")

    with tarfile.open(ARCHIVE_PATH, "r:bz2") as tar:
        tar.extractall(DATA_DIR)

    print("Dataset ready!")

# --------------------------------------------------
# CHARACTER TOKENIZER
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

        # normalized text
        text = row[2]

        wav_path = os.path.join(
            self.root_dir,
            "wavs",
            file_id + ".wav"
        )

        # ----------------------------------------
        # LOAD AUDIO
        # ----------------------------------------

        audio, sr = librosa.load(
            wav_path,
            sr=22050
        )

        # ----------------------------------------
        # MEL SPECTROGRAM
        # ----------------------------------------

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

        # ----------------------------------------
        # TOKENIZE TEXT
        # ----------------------------------------

        text_seq = self.text_to_sequence(text)

        return text_seq, mel

# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":

    # Download dataset
    download_dataset()

    # Load dataset
    dataset = LJDataset(EXTRACTED_PATH)

    print("\nDataset Size:")
    print(len(dataset))

    # First sample
    text_seq, mel = dataset[0]

    print("\nText Sequence Shape:")
    print(text_seq.shape)

    print("\nMel Shape:")
    print(mel.shape)

    print("\nFirst 20 Text Tokens:")
    print(text_seq[:20])
