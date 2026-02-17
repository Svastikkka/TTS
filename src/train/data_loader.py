# data_loader.py
import os
import torch
import librosa
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from phonemizer import phonemize

# Define a simple vocabulary for phonemes
VOCAB = "abcdefghijklmnopqrstuvwxyz. !?, "
CHAR_TO_ID = {char: i for i, char in enumerate(VOCAB)}

class LJSpeechDataset(Dataset):
    def __init__(self, target_dir, n_mels=80):
        self.target_dir = target_dir
        self.wav_dir = os.path.join(target_dir, "wavs")
        self.metadata = pd.read_csv(os.path.join(target_dir, "metadata.csv"), 
                                    sep="|", header=None, quoting=3)
        self.n_mels = n_mels

    def __len__(self):
        return len(self.metadata)

    def preprocess_text(self, text):
        text = text.lower()
        token_ids = [CHAR_TO_ID.get(c, 0) for c in text]
        return torch.LongTensor(token_ids)

    def get_mel(self, filename):
        wav_path = os.path.join(self.wav_dir, f"{filename}.wav")
        audio, _ = librosa.load(wav_path, sr=22050)
        # Standard TTS Mel-spectrogram parameters
        mel = librosa.feature.melspectrogram(y=audio, sr=22050, n_mels=self.n_mels, 
                                             n_fft=1024, hop_length=256)
        mel = librosa.power_to_db(mel, ref=np.max)
        return torch.FloatTensor(mel)

    def __getitem__(self, idx):
        file_id = self.metadata.iloc[idx, 0]
        raw_text = self.metadata.iloc[idx, 2] # Use normalized text
        
        text_tensor = self.preprocess_text(raw_text)
        mel_tensor = self.get_mel(file_id)
        
        return text_tensor, mel_tensor

# A Collate function is necessary because every sentence has a different length
def collate_fn(batch):
    # Padding sequences to match the longest one in the batch
    texts, mels = zip(*batch)
    texts_padded = torch.nn.utils.rnn.pad_sequence(texts, batch_first=True)
    
    # Mel spectrograms need padding on the time axis (last dimension)
    max_mel_len = max([m.size(1) for m in mels])
    mels_padded = torch.stack([torch.nn.functional.pad(m, (0, max_mel_len - m.size(1))) for m in mels])
    
    return texts_padded, mels_padded

if __name__ == "__main__":
    # Test the loader
    dataset = LJSpeechDataset("data/LJSpeech-1.1")
    # On A100, you can easily use batch_size=64 or higher
    loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    
    for texts, mels in loader:
        print(f"Batch Texts Shape: {texts.shape}") # [Batch, Text_Len]
        print(f"Batch Mels Shape: {mels.shape}")   # [Batch, 80, Mel_Len]
        break