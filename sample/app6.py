import torch
import torch.nn as nn

# Example token ids
encoded = [3, 2, 4, 4, 5]

# Convert to tensor
x = torch.tensor(encoded).unsqueeze(0)

print("Input Shape:")
print(x.shape)

# Config
vocab_size = 8
embedding_dim = 16
hidden_dim = 32

# Embedding layer
embedding = nn.Embedding(
    num_embeddings=vocab_size,
    embedding_dim=embedding_dim
)

# LSTM Encoder
encoder = nn.LSTM(
    input_size=embedding_dim,
    hidden_size=hidden_dim,
    batch_first=True
)

# Embedding output
embedded = embedding(x)

print("\nEmbedded Shape:")
print(embedded.shape)

# Encoder output
encoder_output, (hidden, cell) = encoder(embedded)

print("\nEncoder Output Shape:")
print(encoder_output.shape)

print("\nHidden State Shape:")
print(hidden.shape)

print("\nCell State Shape:")
print(cell.shape)