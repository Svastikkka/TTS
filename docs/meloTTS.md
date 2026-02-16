# Melo TTS

- MeloTTS is essentially a refined version of VITS. It combines three core components into one "end-to-end" model:
    1. Text Encoder: Converts text/phonemes into a hidden representation.
    2. Stochastic Duration Predictor: Decides how long each sound (phoneme) should last.
    3. Decoder (Vocoder): Uses a HiFi-GAN based structure to turn those representations directly into a 22kHz or 44kHz waveform.
*Note*: The most effective modern architecture to implement is VITS (Variational Inference with adversarial learning for end-to-end Text-to-Speech) or its successor, VITS2.


### High-Level Data Flow
1. Text Input: Raw text is cleaned and converted into phonemes (using a G2P - Grapheme-to-Phoneme module).
2. Encoder: The phonemes are processed by a Transformer-based encoder to create hidden representations.
3. Duration Predictor: The model predicts how long each phoneme should last to ensure natural rhythm.
4. Flow-based Decoder: This component transforms the simple encoder hidden states into complex, high-dimensional features (latent space) using a series of invertible transformations.
5. Decoder (Vocoder): Finally, a GAN-based generator (similar to HiFi-GAN) converts these latent features directly into a raw audio waveform.

### Key Components
|Component |Function |
|----------|---------|
|BERT Embeddings                |Uses a BERT model to extract semantic information from text, helping the model understand context for better intonation.       |
|Stochastic Duration Predictor  |Predicts the length of speech segments. Being "stochastic" allows for slight variations, making the voice sound less robotic.  |
|Normalizing Flows              |Increases the expressive power of the model, allowing it to generate the complex "texture" of a human voice from simple text.  |
|Multi-Speaker/Accent Layer     |A conditioning layer that allows a single model to switch between accents (e.g., American vs. British English) or languages via a Speaker ID.|
|Adversarial Training           |Uses a Discriminator during training to judge if the audio is "real" or "fake," pushing the Generator to be more realistic.    |

### Training & Testing Resources
Because MeloTTS is an end-to-end model, it is computationally intensive to train but very light to run (test).

Training Requirements
1. GPU: At least 1x NVIDIA GPU with 16GB+ VRAM (e.g., RTX 3090, A100, or V100). Training on 8GB cards is possible by reducing batch_size, but it is significantly slower.

2. CPU/RAM: 16GB+ RAM is recommended to handle data loading and preprocessing.

3. Storage: Depends on your dataset. A typical high-quality dataset (e.g., LJSpeech or VCTK) requires 20GB–50GB of space.

4. Time: Training a model from scratch typically takes 3–7 days on a single high-end GPU. Fine-tuning a pre-trained model on a new voice takes only 2–6 hours.

### Dataset Structure
To train your own version, you need audio files and a metadata file formatted like this: 
```
path/to/audio.wav | speaker_name | language_code | text
```