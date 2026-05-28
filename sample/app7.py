import torch
import torch.nn as nn

# -------------------
# INPUT TOKENS
# -------------------

encoded = [3, 2, 4, 4, 5]

x = torch.tensor(encoded).unsqueeze(0)

# -------------------
# CONFIG
# -------------------

vocab_size = 8
embedding_dim = 16
hidden_dim = 32
mel_dim = 80

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

decoder = nn.LSTM(
    input_size=hidden_dim,
    hidden_size=hidden_dim,
    batch_first=True
)

# -------------------
# MEL PROJECTION
# -------------------

mel_linear = nn.Linear(
    hidden_dim,
    mel_dim
)

# -------------------
# FORWARD PASS
# -------------------

embedded = embedding(x)

encoder_output, (hidden, cell) = encoder(embedded)

print("Encoder Output Shape:")
print(encoder_output.shape)

decoder_output, _ = decoder(encoder_output)

print("\nDecoder Output Shape:")
print(decoder_output.shape)

mel_output = mel_linear(decoder_output)

print("\nMel Output Shape:")
print(mel_output.shape)