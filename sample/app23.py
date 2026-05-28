import os
import numpy as np
import pandas as pd
import librosa
import torch

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# ==================================================
# CONFIG
# ==================================================

DATASET_PATH = "data/LJSpeech-1.1"

SAMPLE_RATE = 22050

N_FFT = 1024
HOP_LENGTH = 256
WIN_LENGTH = 1024
N_MELS = 80

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
# TEXT -> TOKEN IDS
# ==================================================

def text_to_sequence(text):

    text = text.lower()

    sequence = []

    for ch in text:

        if ch in char_to_id:
            sequence.append(
                char_to_id[ch]
            )

    return torch.tensor(sequence)

# ==================================================
# DATASET
# ==================================================

class LJSpeechDataset(Dataset):

    def __init__(self, dataset_path):

        self.dataset_path = dataset_path

        metadata_path = os.path.join(
            dataset_path,
            "metadata.csv"
        )

        self.metadata = pd.read_csv(
            metadata_path,
            sep="|",
            header=None
        )

    def __len__(self):

        return len(self.metadata)

    def __getitem__(self, idx):

        row = self.metadata.iloc[idx]

        file_id = row[0]

        # normalized text
        text = row[2]

        # ------------------------------------------
        # AUDIO PATH
        # ------------------------------------------

        wav_path = os.path.join(
            self.dataset_path,
            "wavs",
            file_id + ".wav"
        )

        # ------------------------------------------
        # LOAD AUDIO
        # ------------------------------------------

        audio, sr = librosa.load(
            wav_path,
            sr=SAMPLE_RATE
        )

        # ------------------------------------------
        # MEL SPECTROGRAM
        # ------------------------------------------

        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            n_mels=N_MELS
        )

        mel = np.log(
            np.clip(
                mel,
                a_min=1e-5,
                a_max=None
            )
        )

        # [80, T] -> [T, 80]
        mel = mel.T

        mel = torch.tensor(
            mel,
            dtype=torch.float32
        )

        # ------------------------------------------
        # TOKENIZE TEXT
        # ------------------------------------------

        tokens = text_to_sequence(text)

        # ------------------------------------------
        # APPROXIMATE DURATIONS
        # ------------------------------------------

        mel_len = mel.shape[0]

        text_len = len(tokens)

        duration = max(
            1,
            mel_len // text_len
        )

        durations = torch.ones(
            text_len
        ) * duration

        durations = durations.long()

        return (
            tokens,
            mel,
            durations
        )

# ==================================================
# COLLATE FUNCTION
# ==================================================

def collate_fn(batch):

    tokens_list = []
    mel_list = []
    durations_list = []

    for item in batch:

        tokens_list.append(item[0])
        mel_list.append(item[1])
        durations_list.append(item[2])

    # ----------------------------------------------
    # FIND MAX LENGTHS
    # ----------------------------------------------

    max_text_len = max(
        len(x)
        for x in tokens_list
    )

    max_mel_len = max(
        x.shape[0]
        for x in mel_list
    )

    # ----------------------------------------------
    # PAD TOKENS
    # ----------------------------------------------

    padded_tokens = []

    for tokens in tokens_list:

        pad_len = max_text_len - len(tokens)

        padded = torch.nn.functional.pad(
            tokens,
            (0, pad_len),
            value=PAD_ID
        )

        padded_tokens.append(padded)

    padded_tokens = torch.stack(
        padded_tokens
    )

    # ----------------------------------------------
    # PAD MELS
    # ----------------------------------------------

    padded_mels = []

    for mel in mel_list:

        pad_len = max_mel_len - mel.shape[0]

        padded = torch.nn.functional.pad(
            mel,
            (0, 0, 0, pad_len)
        )

        padded_mels.append(padded)

    padded_mels = torch.stack(
        padded_mels
    )

    # ----------------------------------------------
    # PAD DURATIONS
    # ----------------------------------------------

    padded_durations = []

    for durations in durations_list:

        pad_len = max_text_len - len(durations)

        padded = torch.nn.functional.pad(
            durations,
            (0, pad_len)
        )

        padded_durations.append(padded)

    padded_durations = torch.stack(
        padded_durations
    )

    return (
        padded_tokens,
        padded_mels,
        padded_durations
    )

# ==================================================
# TEST DATASET
# ==================================================

if __name__ == "__main__":

    dataset = LJSpeechDataset(
        DATASET_PATH
    )

    print("Dataset Size:")
    print(len(dataset))

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn
    )

    batch = next(iter(dataloader))

    tokens, mels, durations = batch

    print("\nTokens Shape:")
    print(tokens.shape)

    print("\nMel Shape:")
    print(mels.shape)

    print("\nDurations Shape:")
    print(durations.shape)
