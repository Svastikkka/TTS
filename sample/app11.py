import torch
import torch.nn as nn
import torch.optim as optim

# -------------------
# CONFIG
# -------------------

batch_size = 1
seq_len = 20
mel_dim = 80
hidden_dim = 32

# -------------------
# FAKE TARGET MEL
# -------------------

target_mel = torch.randn(
    batch_size,
    seq_len,
    mel_dim
)

print("Target Mel Shape:")
print(target_mel.shape)

# -------------------
# DECODER
# -------------------

decoder = nn.LSTMCell(
    input_size=mel_dim,
    hidden_size=hidden_dim
)

mel_projection = nn.Linear(
    hidden_dim,
    mel_dim
)

# -------------------
# OPTIMIZER
# -------------------

optimizer = optim.Adam(
    list(decoder.parameters()) +
    list(mel_projection.parameters()),
    lr=0.001
)

# -------------------
# LOSS
# -------------------

criterion = nn.MSELoss()

# -------------------
# INITIAL STATES
# -------------------

hidden = torch.zeros(batch_size, hidden_dim)
cell = torch.zeros(batch_size, hidden_dim)

# Initial input frame
prev_frame = torch.zeros(batch_size, mel_dim)

generated_frames = []

# -------------------
# AUTOREGRESSIVE TRAINING
# -------------------

for t in range(seq_len):

    hidden, cell = decoder(
        prev_frame,
        (hidden, cell)
    )

    predicted_frame = mel_projection(hidden)

    generated_frames.append(predicted_frame)

    # -------------------------
    # TEACHER FORCING
    # -------------------------

    prev_frame = target_mel[:, t, :]

# -------------------
# STACK OUTPUT
# -------------------

generated_mel = torch.stack(
    generated_frames,
    dim=1
)

print("\nGenerated Mel Shape:")
print(generated_mel.shape)

# -------------------
# COMPUTE LOSS
# -------------------

loss = criterion(
    generated_mel,
    target_mel
)

print("\nLoss:")
print(loss.item())

# -------------------
# BACKPROP
# -------------------

optimizer.zero_grad()

loss.backward()

optimizer.step()

print("\nTraining Step Complete")