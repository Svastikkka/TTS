import torch
import torch.nn as nn
import torch.nn.functional as F

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

# ------------------------------------------------
# ATTENTION
# ------------------------------------------------

# decoder_output:
# [batch, decoder_time, hidden_dim]

# encoder_output:
# [batch, encoder_time, hidden_dim]

attention_scores = torch.bmm(
    decoder_output,
    encoder_output.transpose(1, 2)
)

print("\nAttention Scores Shape:")
print(attention_scores.shape)

# Softmax → probabilities
attention_weights = F.softmax(
    attention_scores,
    dim=-1
)

print("\nAttention Weights Shape:")
print(attention_weights.shape)

# Context vector
context = torch.bmm(
    attention_weights,
    encoder_output
)

print("\nContext Shape:")
print(context.shape)

# Final mel prediction
mel_output = mel_linear(context)

print("\nMel Output Shape:")
print(mel_output.shape)