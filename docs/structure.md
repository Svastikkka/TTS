```
tts-engine/
├── README.md
├── requirements.txt
├── config/
│   ├── model_config.yaml        # Model paths, ONNX/TensorRT configs
│   ├── normalization_rules.yaml # Custom text normalization rules
│   └── voices.yaml              # Voice and language mapping
├── data/
│   ├── phoneme_dicts/           # G2P dictionaries per language (optional)
│   ├── test_sentences.txt       # Sample input for benchmarking
│   └── ...
├── src/
│   ├── __init__.py
│   ├── main.py                  # Entry point (CLI or API)
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── server.py            # REST or gRPC/WebSocket server
│   │   └── routes.py            # API endpoints
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── language_detector.py # langdetect / fastText wrapper
│   │   ├── normalizer.py        # Regex + rule-based text normalization
│   │   ├── phonemizer.py        # G2P (phoneme generator)
│   │   ├── acoustic_model.py    # TensorRT inference for FastSpeech2
│   │   ├── vocoder.py           # TensorRT inference for HiFi-GAN
│   │   └── pipeline.py          # End-to-end pipeline orchestration
│   │
│   ├── utils/
│   │   ├── logger.py            # Unified logging
│   │   ├── profiler.py          # Latency measurement tools
│   │   └── audio.py             # Audio utilities (save, stream, format)
│   │
│   └── benchmarks/
│       ├── benchmark.py         # Script to measure latency, throughput
│       └── results/             # Benchmark results & logs
│
├── notebooks/
│   ├── training.ipynb           # Model training / fine-tuning
│   ├── export_to_onnx.ipynb     # Convert PyTorch model → ONNX
│   └── optimize_trt.ipynb       # TensorRT engine building & testing
│
└── tests/
    ├── test_normalizer.py
    ├── test_pipeline.py
    └── test_latency.py

```
