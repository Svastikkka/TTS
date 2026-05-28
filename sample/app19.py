import torch
import torch.nn as nn

# ==================================================
# FAKE ENCODER OUTPUT
# ==================================================

# [batch, seq_len, hidden_dim]

encoder_output = torch.randn(
    1,
    5,
    256
)

print("Encoder Output Shape:")
print(encoder_output.shape)

# ==================================================
# DURATION PREDICTOR
# ==================================================

class DurationPredictor(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels=256,
            out_channels=256,
            kernel_size=3,
            padding=1
        )

        self.relu1 = nn.ReLU()

        self.layer_norm1 = nn.LayerNorm(256)

        self.conv2 = nn.Conv1d(
            in_channels=256,
            out_channels=256,
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

        # [B, T, H]
        # -> [B, H, T]

        x = x.transpose(1, 2)

        x = self.conv1(x)

        x = self.relu1(x)

        # Back to [B, T, H]
        x = x.transpose(1, 2)

        x = self.layer_norm1(x)

        # Again for conv
        x = x.transpose(1, 2)

        x = self.conv2(x)

        x = self.relu2(x)

        x = x.transpose(1, 2)

        x = self.layer_norm2(x)

        durations = self.linear(x)

        # Remove last dimension
        durations = durations.squeeze(-1)

        return durations

# ==================================================
# MODEL
# ==================================================

predictor = DurationPredictor()

predicted_durations = predictor(
    encoder_output
)

print("\nPredicted Durations Shape:")
print(predicted_durations.shape)

print("\nPredicted Durations:")
print(predicted_durations)
