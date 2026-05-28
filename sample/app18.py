import torch

# ==================================================
# EXAMPLE ENCODER OUTPUT
# ==================================================

# [tokens, hidden_dim]

encoder_output = torch.tensor([
    [1.0, 1.0],  # h
    [2.0, 2.0],  # e
    [3.0, 3.0],  # l
    [4.0, 4.0]   # o
])

print("Encoder Output:")
print(encoder_output)

print("\nShape:")
print(encoder_output.shape)

# ==================================================
# DURATIONS
# ==================================================

durations = torch.tensor([
    2,  # h repeated 2 times
    3,  # e repeated 3 times
    4,  # l repeated 4 times
    2   # o repeated 2 times
])

print("\nDurations:")
print(durations)

# ==================================================
# LENGTH REGULATOR
# ==================================================

expanded = []

for i, duration in enumerate(durations):

    repeated = encoder_output[i].unsqueeze(0).repeat(
        duration,
        1
    )

    expanded.append(repeated)

expanded_output = torch.cat(expanded, dim=0)

# ==================================================
# RESULTS
# ==================================================

print("\nExpanded Output:")
print(expanded_output)

print("\nExpanded Shape:")
print(expanded_output.shape)
