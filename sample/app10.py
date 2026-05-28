import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# -------------------
# TEXT INPUT
# -------------------

text = "hello"

chars = sorted(list(set(text)))

char_to_id = {ch: i for i, ch in enumerate(chars)}

encoded = [char_to_id[ch] for ch in text]

x = torch.tensor(encoded).unsqueeze(0)

# -------------------
# CONFIG
# -------------------

vocab_size = len(chars)
embedding_dim = 16
hidden_dim = 32
mel_dim = 80

decoder_steps = 20

# -------------------
# EMBEDDING
# -------------------

embedding = nn.Embedding(
    vocab_size,
    embedding_dim
)

# -------------------
# ENCODER
# -------------------

encoder = nn.LSTM(
    input_size=embedding_dim,
    hidden_size=hidden_dim,
    batch_first=True
)

# -------------------
# DECODER
# -------------------

decoder = nn.LSTMCell(
    input_size=mel_dim + hidden_dim,
    hidden_size=hidden_dim
)

# -------------------
# MEL PROJECTION
# -------------------

mel_linear = nn.Linear(
    hidden_dim,
    mel_dim
)

# -------------------
# ENCODE TEXT
# -------------------

embedded = embedding(x)

encoder_output, _ = encoder(embedded)

# -------------------
# INITIAL STATES
# -------------------

batch_size = 1

hidden = torch.zeros(batch_size, hidden_dim)
cell = torch.zeros(batch_size, hidden_dim)

# Initial mel frame
prev_mel = torch.zeros(batch_size, mel_dim)

# Store generated frames
generated_mels = []

# -------------------
# AUTOREGRESSIVE LOOP
# -------------------

for step in range(decoder_steps):

    # Simple attention:
    # average encoder outputs
    context = encoder_output.mean(dim=1)

    # Decoder input
    decoder_input = torch.cat(
        [prev_mel, context],
        dim=-1
    )

    # Decoder step
    hidden, cell = decoder(
        decoder_input,
        (hidden, cell)
    )

    # Predict mel frame
    mel_frame = mel_linear(hidden)

    # Store frame
    generated_mels.append(mel_frame)

    # Feedback loop
    prev_mel = mel_frame

# -------------------
# STACK FRAMES
# -------------------

mel_output = torch.stack(
    generated_mels,
    dim=1
)

print("Generated Mel Shape:")
print(mel_output.shape)

# -------------------
# VISUALIZE
# -------------------

mel_image = mel_output[0].detach().numpy().T

plt.figure(figsize=(10, 4))

plt.imshow(
    mel_image,
    aspect='auto',
    origin='lower'
)

plt.colorbar()
plt.title("Generated Mel Spectrogram")
plt.xlabel("Time")
plt.ylabel("Mel Channels")

plt.show()