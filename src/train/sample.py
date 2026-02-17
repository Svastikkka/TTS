import torch
import torch.nn as nn

class MeloLikeTTS(nn.Module):
    def __init__(self, vocab_size=50, num_speakers=10, speaker_dim=256, n_mels=80):
        super().__init__()

        self.text_encoder = nn.Embedding(vocab_size, 256)
        self.speaker_emb = nn.Embedding(num_speakers, speaker_dim)

        self.decoder = nn.ConvTranspose1d(
            256 + speaker_dim,
            n_mels,
            kernel_size=256,
            stride=128
        )

    def forward(self, text_seq, speaker_id):
        t_feat = self.text_encoder(text_seq).transpose(1, 2)
        s_feat = self.speaker_emb(speaker_id).unsqueeze(-1)
        s_feat = s_feat.expand(-1, -1, t_feat.size(-1))
        combined = torch.cat([t_feat, s_feat], dim=1)

        return self.decoder(combined)