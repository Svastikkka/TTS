# save as train.py
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader

# Import your previous classes
from sample import MeloLikeTTS
from train.data_loader import LJSpeechDataset, collate_fn

# 1. Hyperparameters Optimized for A100
BATCH_SIZE = 32  # You can go up to 128 on an 80GB A100
LEARNING_RATE = 2e-4
EPOCHS = 100
DEVICE = "cuda"

def train():
    # 2. Initialize Dataset and Loader
    dataset = LJSpeechDataset("data/LJSpeech-1.1")
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, 
                        collate_fn=collate_fn, num_workers=4, pin_memory=True)

    # 3. Initialize Model, Optimizer, and Scaler
    # vocab_size 50 covers our basic alphabet + phonemes
    model = MeloLikeTTS(vocab_size=50, num_speakers=1, speaker_dim=256).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = GradScaler('cuda')
    criterion = nn.MSELoss()

    print(f"Starting training on {torch.cuda.get_device_name(0)}...")

    for epoch in range(EPOCHS):
        total_loss = 0
        for i, (texts, mels) in enumerate(loader):
            texts, mels = texts.to(DEVICE), mels.to(DEVICE)
            
            # Since LJSpeech is 1 speaker, we use Speaker ID 0 for all
            speaker_ids = torch.zeros(texts.size(0), dtype=torch.long).to(DEVICE)

            optimizer.zero_grad()

            # Mixed Precision Forward Pass
            with autocast('cuda', dtype=torch.bfloat16):
                # Model predicts Mel-spectrogram
                # Shape adjustment: we need to make sure model output matches mels
                output = model(texts, speaker_ids)
                
                # Trim or pad output to match ground truth mel length
                output = output[:, :, :mels.size(2)]
                loss = criterion(output, mels)

            # Backward Pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            if i % 10 == 0:
                print(f"Epoch {epoch} | Step {i} | Loss: {loss.item():.4f}")

        # Save Checkpoint
        torch.save(model.state_state_dict(), f"melo_model_epoch_{epoch}.pth")
        print(f"--- Epoch {epoch} Average Loss: {total_loss/len(loader):.4f} ---")

if __name__ == "__main__":
    train()