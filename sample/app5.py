import torch
import torch.nn as nn

# Example encoded text
encoded = [3, 2, 4, 4, 5]

# Convert to tensor
x = torch.tensor(encoded)

print("Input Tensor:")
print(x)

# Vocabulary size
vocab_size = 8

# Embedding dimension
embedding_dim = 16

# Create embedding layer
embedding = nn.Embedding(
    num_embeddings=vocab_size,
    embedding_dim=embedding_dim
)

# Pass tokens through embedding
output = embedding(x)

print("\nEmbedding Output Shape:")
print(output.shape)

print("\nEmbedding Output:")
print(output)