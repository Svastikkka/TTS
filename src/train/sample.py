# save it as sample.py
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler

class MeloLikeTTS(nn.Module):
    def __init__(self, vocab_size=50, num_speakers=10, speaker_dim=256):
        super().__init__()
        # 1. Text representation
        self.text_encoder = nn.Embedding(vocab_size, 256)
        
        # 2. Speaker representation (The "Voice Fingerprint")
        self.speaker_emb = nn.Embedding(num_speakers, speaker_dim)
        
        # 3. Fusion & Decoder
        self.decoder = nn.ConvTranspose1d(256 + speaker_dim, 1, 256, stride=128)

    def forward(self, text_seq, speaker_id):
        # text_seq: [Batch, Length]
        # speaker_id: [Batch]
        
        t_feat = self.text_encoder(text_seq).transpose(1, 2) # [B, 256, T]
        s_feat = self.speaker_emb(speaker_id).unsqueeze(-1) # [B, 256, 1]
        
        # Broadcast speaker features across the entire time dimension
        s_feat = s_feat.expand(-1, -1, t_feat.size(-1))
        
        # Combine Text + Speaker
        combined = torch.cat([t_feat, s_feat], dim=1) # [B, 512, T]
        
        return self.decoder(combined)

# Setup for A100
device = "cuda"
model = MeloLikeTTS(num_speakers=100).to(device) # Support 100 voices
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scaler = GradScaler('cuda')

def train_step(text_batch, speaker_ids):
    model.train()
    optimizer.zero_grad()
    
    with autocast('cuda', dtype=torch.bfloat16):
        # Fake data for demonstration
        txt = torch.randint(0, 50, (1, 20)).to(device)
        spk = torch.tensor([speaker_ids]).to(device)
        
        output = model(txt, spk)
        loss = output.pow(2).mean() # Simple MSE Loss to move toward zero
        
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    print(f"Step Loss: {loss.item():.4f} | Speaker ID: {speaker_ids}")

if __name__ == "__main__":
    train_step("Hello World", speaker_ids=7) # Train with Speaker #7