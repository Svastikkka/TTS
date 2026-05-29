import os
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import Subset


if torch.cuda.is_available():
    device = "cuda:1" 
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print("Using Device:", device)

DATASET_PATH = "data/LJSpeech-1.1"
SAMPLE_RATE = 22050
N_FFT = 1024
HOP_LENGTH = 256
WIN_LENGTH = 1024
N_MELS = 80
BATCH_SIZE = 32
PAD_ID = 0


characters = "abcdefghijklmnopqrstuvwxyz!'?,. "

char_to_id = {
    ch: i + 1
    for i, ch in enumerate(characters)
}

VOCAB_SIZE = len(char_to_id) + 1

def text_to_sequence(text):

    text = str(text).strip().lower()

    if text == "nan":
        text = ""

    sequence = []

    for ch in text:

        if ch in char_to_id:

            sequence.append(
                char_to_id[ch]
            )

    return torch.tensor(
        sequence,
        dtype=torch.long,
        device=device
    )

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

        # Safe text conversion
        text = str(row[2]).strip()

        if text == "nan":
            text = ""

        wav_path = os.path.join(
            self.dataset_path,
            "wavs",
            file_id + ".wav"
        )

        audio, sr = librosa.load(
            wav_path,
            sr=SAMPLE_RATE
        )

        audio_tensor = torch.tensor(audio, dtype=torch.float32, device=device)
        window = torch.hann_window(WIN_LENGTH, device=device)
        
        stft = torch.stft(
            audio_tensor,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            window=window,
            center=True,
            return_complex=True
        )
        
        magnitudes = torch.abs(stft) ** 2
        
        fb = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS)
        fb_tensor = torch.tensor(fb, dtype=torch.float32, device=device)
        
        mel = torch.matmul(fb_tensor, magnitudes)
        mel = torch.log10(torch.clamp(mel, min=1e-5)) * 3.2582
        mel = mel.T

        tokens = text_to_sequence(text)

        mel_len = mel.shape[0]

        text_len = len(tokens)

        base_duration = mel_len // max(text_len, 1)

        remainder = mel_len % max(text_len, 1)

        durations = torch.ones(
            text_len,
            dtype=torch.long,
            device=device
        ) * base_duration

        durations[:remainder] += 1

        return (
            tokens,
            mel,
            durations
        )

def collate_fn(batch):

    tokens_list = []
    mel_list = []
    durations_list = []

    for item in batch:

        tokens_list.append(item[0])
        mel_list.append(item[1])
        durations_list.append(item[2])

    max_text_len = max(
        len(x)
        for x in tokens_list
    )

    max_mel_len = max(
        x.shape[0]
        for x in mel_list
    )

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

class LengthRegulator(nn.Module):

    def __init__(self):

        super().__init__()

    def forward(
        self,
        encoder_output,
        durations
    ):

        expanded_batch = []

        batch_size = encoder_output.shape[0]

        for b in range(batch_size):

            expanded = []

            for i, duration in enumerate(durations[b]):

                duration = max(
                    1,
                    int(duration.item())
                )

                repeated = encoder_output[b, i].unsqueeze(0).repeat(
                    duration,
                    1
                )

                expanded.append(repeated)

            expanded = torch.cat(
                expanded,
                dim=0
            )

            expanded_batch.append(expanded)

        max_len = max(
            x.shape[0]
            for x in expanded_batch
        )

        padded_batch = []

        for x in expanded_batch:

            pad_len = max_len - x.shape[0]

            padded = nn.functional.pad(
                x,
                (0, 0, 0, pad_len)
            )

            padded_batch.append(padded)

        return torch.stack(
            padded_batch
        )

class DurationPredictor(nn.Module):

    def __init__(self, input_dim=512): 

        super().__init__()

        self.conv1 = nn.Conv1d(
            input_dim,
            256,
            kernel_size=3,
            padding=1
        )

        self.relu1 = nn.ReLU()

        self.layer_norm1 = nn.LayerNorm(256)

        self.conv2 = nn.Conv1d(
            256,
            256,
            kernel_size=3,
            padding=1
        )

        self.relu2 = nn.ReLU()

        self.layer_norm2 = nn.LayerNorm(256)

        self.linear = nn.Linear(
            256,
            1
        )

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

        durations = torch.relu(
            durations
        )

        durations = durations.squeeze(-1)

        return durations

class MiniFastSpeech(nn.Module):

    def __init__(self):

        super().__init__()

        self.embedding = nn.Embedding(
            VOCAB_SIZE,
            256, 
            padding_idx=PAD_ID
        )

        self.encoder = nn.LSTM(
            input_size=256, 
            hidden_size=256, 
            num_layers=2, 
            batch_first=True,
            bidirectional=True
        )

        self.duration_predictor = DurationPredictor(input_dim=512) 

        self.length_regulator = LengthRegulator()

        self.decoder = nn.LSTM(
            input_size=512, 
            hidden_size=256, 
            num_layers=2, 
            batch_first=True
        )

        self.mel_projection = nn.Linear(
            256,
            N_MELS
        )

    def forward(
        self,
        text,
        durations=None
    ):

        x = self.embedding(text)

        x, _ = self.encoder(x)

        predicted_durations = self.duration_predictor(x)

        if durations is None:

            durations = torch.round(
                predicted_durations
            ).long()

            durations = torch.clamp(
                durations,
                min=1
            )

        x = self.length_regulator(
            x,
            durations
        )

        # ------------------------------------------------
        # DECODER
        # ------------------------------------------------

        x, _ = self.decoder(x)

        # ------------------------------------------------
        # MEL OUTPUT
        # ------------------------------------------------

        mel = self.mel_projection(x)

        return (
            mel,
            predicted_durations
        )


# ==================================================
# MAIN EXECUTION ENTRY BLOCK
# ==================================================

if __name__ == "__main__":
    # Fixes the CUDA multi-processing error by setting start method to 'spawn'
    import multiprocessing
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass 

    # ==================================================
    # DATASET Setup
    # ==================================================

    dataset = LJSpeechDataset(
        DATASET_PATH
    )

    # REMOVED Subset constraint parameters entirely to pipe the complete dataset
    dataloader = DataLoader(
        dataset,                  # Directly loading entire production dataset
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=False
    )

    print("Complete Production Dataset Size:", len(dataset))

    # ==================================================
    # MODEL
    # ==================================================

    model = MiniFastSpeech().to(device)

    # SPEEDUP: Compile the model layout for specialized Lovelace architecture optimizations
    if hasattr(torch, "compile") and device != "cpu":
        print("Compiling model graph via torch.compile() for structural optimization...")
        compiled_model = torch.compile(model)
    else:
        compiled_model = model

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001
    )

    mel_loss_fn = nn.MSELoss()

    duration_loss_fn = nn.MSELoss()

    # SPEEDUP: Instantiate the Gradient Scaler for Automatic Mixed Precision (AMP)
    scaler = torch.amp.GradScaler("cuda" if "cuda" in device else "cpu")

    # ==================================================
    # RESUME CHECKPOINT LOGIC
    # ==================================================
    CHECKPOINT_PATH = "fastspeech_checkpoint.pth"
    start_epoch = 0

    if os.path.exists(CHECKPOINT_PATH):
        print(f"Found existing checkpoint at '{CHECKPOINT_PATH}'. Loading state...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Resuming training directly from Epoch {start_epoch + 1}!")
    else:
        print("No prior checkpoint discovered. Starting fresh training session.")

    # ==================================================
    # TRAINING
    # ==================================================

    epochs = 100 # Swapped from 10 to 100 full structural dataset cycles

    for epoch in range(start_epoch, epochs):

        total_loss = 0

        for batch_idx, batch in enumerate(dataloader):

            tokens, target_mels, target_durations = batch

            # ------------------------------------------------
            # MOVE TO TARGET GPU (Isolated Device 1)
            # ------------------------------------------------

            tokens = tokens.to(
                device,
                non_blocking=True
            )

            target_mels = target_mels.to(
                device,
                non_blocking=True
            )

            target_durations = target_durations.to(
                device,
                non_blocking=True
            )

            # ------------------------------------------------
            # FORWARD & BACKPROP WITH AUTOMATIC MIXED PRECISION (AMP)
            # ------------------------------------------------
            optimizer.zero_grad()

            # Enforce dynamic Bfloat16/Float16 execution context depending on target hardware configuration
            amp_device_type = "cuda" if "cuda" in device else "cpu"
            with torch.amp.autocast(device_type=amp_device_type, dtype=torch.bfloat16):
                predicted_mels, predicted_durations = compiled_model(
                    tokens,
                    target_durations
                )

                # ------------------------------------------------
                # MATCH LENGTHS
                # ------------------------------------------------

                min_len = min(
                    predicted_mels.shape[1],
                    target_mels.shape[1]
                )

                predicted_mels = predicted_mels[:, :min_len]

                target_mels = target_mels[:, :min_len]

                # ------------------------------------------------
                # LOSSES
                # ------------------------------------------------

                mel_loss = mel_loss_fn(
                    predicted_mels,
                    target_mels
                )

                duration_loss = duration_loss_fn(
                    predicted_durations,
                    target_durations.float()
                )

                loss = mel_loss + duration_loss

            # Scale the loss and complete backward pass via AMP custom pipelines
            scaler.scale(loss).backward()

            # Unscale the gradients before clipping to ensure tracking stays precise
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )

            # Step optimizer and update scale factors
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            # ------------------------------------------------
            # PRINT
            # ------------------------------------------------
            
            # Reduced print clutter by reporting every 50 batches instead of every single step
            if batch_idx % 50 == 0:
                print(
                    f"Epoch {epoch+1}/{epochs} | "
                    f"Batch {batch_idx}/{len(dataloader)} | "
                    f"Loss: {loss.item():.4f}"
                )

        avg_loss = total_loss / len(dataloader)

        print(
            f"\n--- Epoch {epoch+1} Complete. Average Dataset Loss: "
            f"{avg_loss:.4f} ---\n"
        )

        # ---------------- ==============================
        # SAVE EVERY EPOCH COMPLETE CHECKPOINT
        # ---------------- ==============================
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, CHECKPOINT_PATH)
        print(f"Checkpoint successfully synchronized and saved to '{CHECKPOINT_PATH}'")

    # ==================================================
    # SAVE MODEL
    # ==================================================

    torch.save(
        model.state_dict(),
        "mini_fastspeech.pth"
    )

    print("Model Saved!")