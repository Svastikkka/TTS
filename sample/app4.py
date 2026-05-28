text = "hello world"

# Create vocabulary
chars = sorted(list(set(text)))

print("Vocabulary:", chars)

# Create mappings
char_to_id = {ch: i for i, ch in enumerate(chars)}
id_to_char = {i: ch for ch, i in char_to_id.items()}

print("\nChar to ID:")
print(char_to_id)

# Encode text
encoded = [char_to_id[ch] for ch in text]

print("\nEncoded:")
print(encoded)

# Decode back
decoded = ''.join([id_to_char[i] for i in encoded])

print("\nDecoded:")
print(decoded)