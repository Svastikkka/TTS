Why these specific versions?
- PyTorch 2.4.0 + cu124: This version includes optimized kernels for the Ampere architecture (A100). It allows you to use torch.compile, which can speed up your TTS training by up to 30%.
- Phonemizer 3.2.1: This is the industry standard for G2P (Grapheme-to-Phoneme). Note that it requires espeak-ng to be installed on your system (sudo apt-get install espeak-ng on Ubuntu).
- A100 Strategy: Since you have 80GB of VRAM, these versions support Flash Attention 2, which is critical if you decide to use a Transformer-based encoder for your TTS model.