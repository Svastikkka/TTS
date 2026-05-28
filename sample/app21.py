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
# FAKE ENCODER OUTPUT
# ==================================================

batch_size = 2
seq_len = 10
hidden_dim = 256

encoder_output = torch.randn(
    batch_size,
    seq_len,
    hidden_dim
).to(device)

print("Encoder Output Shape:")
print(encoder_output.shape)

# ==================================================
# FAKE MEL LENGTHS
# ==================================================

mel_lengths = torch.tensor([
    50,
    80
]).to(device)

print("\nMel Lengths:")
print(mel_lengths)

# ==================================================
# CREATE TARGET DURATIONS
# ==================================================

target_durations = []

for mel_len in mel_lengths:

    # Uniform approximation
    duration = mel_len.item() // seq_len

    durations = torch.ones(
        seq_len
    ) * duration

    target_durations.append(durations)

target_durations = torch.stack(
    target_durations
).to(device)

print("\nTarget Durations:")
print(target_durations)

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
# MODEL
# ==================================================

model = DurationPredictor().to(device)

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

criterion = nn.MSELoss()

# ==================================================
# TRAINING LOOP
# ==================================================

epochs = 10

for epoch in range(epochs):

    predicted_durations = model(
        encoder_output
    )

    loss = criterion(
        predicted_durations,
        target_durations
    )

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    print(
        f"Epoch {epoch+1} "
        f"Loss: {loss.item():.4f}"
    )

# ==================================================
# FINAL PREDICTIONS
# ==================================================

print("\nFinal Predicted Durations:")

print(
    predicted_durations.detach()
)
