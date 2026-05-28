import torch
import torch.nn as nn
import torch.optim as optim

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
# CONFIG
# ==================================================

VOCAB_SIZE = 100
PAD_ID = 0

batch_size = 2
text_len = 10
mel_len = 50
mel_dim = 80

# ==================================================
# FAKE DATA
# ==================================================

tokens = torch.randint(
    1,
    VOCAB_SIZE,
    (batch_size, text_len)
).to(device)

target_mels = torch.randn(
    batch_size,
    mel_len,
    mel_dim
).to(device)

print("Tokens Shape:")
print(tokens.shape)

print("\nTarget Mel Shape:")
print(target_mels.shape)

# ==================================================
# CREATE TARGET DURATIONS
# ==================================================

target_durations = []

for _ in range(batch_size):

    duration = mel_len // text_len

    durations = torch.ones(
        text_len
    ) * duration

    target_durations.append(durations)

target_durations = torch.stack(
    target_durations
).to(device)

print("\nTarget Durations:")
print(target_durations)

# ==================================================
# LENGTH REGULATOR
# ==================================================

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

        # --------------------------------------
        # PAD TO SAME LENGTH
        # --------------------------------------

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

        return torch.stack(padded_batch)

# ==================================================
# DURATION PREDICTOR
# ==================================================

class DurationPredictor(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1 = nn.Conv1d(
            256,
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

# ==================================================
# FASTSPEECH MODEL
# ==================================================

class MiniFastSpeech(nn.Module):

    def __init__(self):

        super().__init__()

        # --------------------------------------
        # EMBEDDING
        # --------------------------------------

        self.embedding = nn.Embedding(
            VOCAB_SIZE,
            128,
            padding_idx=PAD_ID
        )

        # --------------------------------------
        # ENCODER
        # --------------------------------------

        self.encoder = nn.LSTM(
            input_size=128,
            hidden_size=128,
            batch_first=True,
            bidirectional=True
        )

        # --------------------------------------
        # DURATION PREDICTOR
        # --------------------------------------

        self.duration_predictor = DurationPredictor()

        # --------------------------------------
        # LENGTH REGULATOR
        # --------------------------------------

        self.length_regulator = LengthRegulator()

        # --------------------------------------
        # DECODER
        # --------------------------------------

        self.decoder = nn.LSTM(
            input_size=256,
            hidden_size=256,
            batch_first=True
        )

        # --------------------------------------
        # MEL PROJECTION
        # --------------------------------------

        self.mel_projection = nn.Linear(
            256,
            mel_dim
        )

    def forward(
        self,
        text,
        target_durations
    ):

        # --------------------------------------
        # EMBEDDING
        # --------------------------------------

        x = self.embedding(text)

        # --------------------------------------
        # ENCODER
        # --------------------------------------

        x, _ = self.encoder(x)

        # --------------------------------------
        # DURATION PREDICTOR
        # --------------------------------------

        predicted_durations = self.duration_predictor(x)

        # --------------------------------------
        # LENGTH REGULATOR
        # --------------------------------------

        x = self.length_regulator(
            x,
            target_durations
        )

        # --------------------------------------
        # DECODER
        # --------------------------------------

        x, _ = self.decoder(x)

        # --------------------------------------
        # MEL OUTPUT
        # --------------------------------------

        mel = self.mel_projection(x)

        return (
            mel,
            predicted_durations
        )

# ==================================================
# MODEL
# ==================================================

model = MiniFastSpeech().to(device)

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

mel_loss_fn = nn.MSELoss()

duration_loss_fn = nn.MSELoss()

# ==================================================
# TRAINING LOOP
# ==================================================

epochs = 10

for epoch in range(epochs):

    predicted_mels, predicted_durations = model(
        tokens,
        target_durations
    )

    # --------------------------------------
    # MEL LOSS
    # --------------------------------------

    mel_loss = mel_loss_fn(
        predicted_mels,
        target_mels
    )

    # --------------------------------------
    # DURATION LOSS
    # --------------------------------------

    duration_loss = duration_loss_fn(
        predicted_durations,
        target_durations
    )

    # --------------------------------------
    # TOTAL LOSS
    # --------------------------------------

    total_loss = (
        mel_loss +
        duration_loss
    )

    optimizer.zero_grad()

    total_loss.backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        1.0
    )

    optimizer.step()

    print(
        f"Epoch {epoch+1} | "
        f"Mel Loss: {mel_loss.item():.4f} | "
        f"Duration Loss: {duration_loss.item():.4f}"
    )
