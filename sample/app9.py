import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# -------------------
# INPUT TOKENS
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
# FORWARD
# -------------------

embedded = embedding(x)

encoder_output, _ = encoder(embedded)

decoder_output, _ = decoder(encoder_output)

# -------------------
# ATTENTION
# -------------------

attention_scores = torch.bmm(
    decoder_output,
    encoder_output.transpose(1, 2)
)

attention_weights = F.softmax(
    attention_scores,
    dim=-1
)

# Remove batch dimension
attention_map = attention_weights[0].detach().numpy()

# -------------------
# VISUALIZATION
# -------------------

plt.figure(figsize=(6, 5))

plt.imshow(
    attention_map,
    aspect='auto',
    origin='lower'
)

plt.colorbar()

plt.xticks(range(len(text)), list(text))
plt.yticks(range(len(text)))

plt.xlabel("Encoder Tokens")
plt.ylabel("Decoder Steps")
plt.title("Attention Alignment")

plt.show()