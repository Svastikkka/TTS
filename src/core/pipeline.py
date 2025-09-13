import numpy as np
from src.core.language_detector import detect_language
from src.core.normalizer import normalize_text
from src.core.phonemizer import phonemize_text
from src.core.acoustic_model import AcousticModelTRT
from src.core.vocoder import VocoderTRT

class TTSPipeline:
    def __init__(self):
        # Load TensorRT engines once during startup
        self.acoustic_model = AcousticModelTRT("config/model_config.yaml")
        self.vocoder = VocoderTRT("config/model_config.yaml")

    def run(self, text: str, language: str):
        # Step 1: Language Detection (optional)
        lang = language or detect_language(text)

        # Step 2: Text Normalization
        normalized = normalize_text(text, lang)

        # Step 3: Phoneme Generation
        phonemes = phonemize_text(normalized, lang)

        # Step 4: Acoustic Model (TRT)
        mel_spectrogram = self.acoustic_model.infer(phonemes)

        # Step 5: Vocoder (TRT)
        audio_waveform = self.vocoder.infer(mel_spectrogram)

        return audio_waveform, self.vocoder.sample_rate
