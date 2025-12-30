import torch
import phonemizer

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"GPU Name: {torch.cuda.get_device_name(0)}")
print(f"Phonemizer Version: {phonemizer.__version__}")

# Check if A100 is utilizing BFloat16 (Great for MeloTTS-like training)
if torch.cuda.is_bf16_supported():
    print("BFloat16 is supported! (Recommended for training on A100)")