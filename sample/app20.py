import torch
import torch.nn as nn

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
# TOKENIZER CONFIG
# ==================================================

VOCAB_SIZE = 100
PAD_ID = 0

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

        # ------------------------------------------
        # PAD TO SAME LENGTH
        # ------------------------------------------

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
# MINI FASTSPEECH
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
            80
        )

    def forward(
        self,
        text,
        durations=None
    ):

        # --------------------------------------
        # EMBEDDING
        # --------------------------------------

        x = self.embedding(text)

        print("\nEmbedding Shape:")
        print(x.shape)

        # --------------------------------------
        # ENCODER
        # --------------------------------------

        x, _ = self.encoder(x)

        print("\nEncoder Shape:")
        print(x.shape)

        # --------------------------------------
        # DURATION PREDICTOR
        # --------------------------------------

        predicted_durations = self.duration_predictor(x)

        print("\nPredicted Durations Shape:")
        print(predicted_durations.shape)

        # --------------------------------------
        # USE FAKE DURATIONS FOR NOW
        # --------------------------------------

        if durations is None:

            durations = torch.ones_like(
                predicted_durations
            ) * 5

        # --------------------------------------
        # LENGTH REGULATOR
        # --------------------------------------

        x = self.length_regulator(
            x,
            durations
        )

        print("\nExpanded Shape:")
        print(x.shape)

        # --------------------------------------
        # DECODER
        # --------------------------------------

        x, _ = self.decoder(x)

        print("\nDecoder Shape:")
        print(x.shape)

        # --------------------------------------
        # MEL OUTPUT
        # --------------------------------------

        mel = self.mel_projection(x)

        print("\nMel Shape:")
        print(mel.shape)

        return mel

# ==================================================
# TEST MODEL
# ==================================================

model = MiniFastSpeech().to(device)

# Fake token batch
tokens = torch.randint(
    1,
    VOCAB_SIZE,
    (2, 10)
).to(device)

print("Input Tokens Shape:")
print(tokens.shape)

# Forward pass
mel = model(tokens)
