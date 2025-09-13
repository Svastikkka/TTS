✅ Base Image
- Using nvcr.io/nvidia/pytorch ensures:
- CUDA, cuDNN, TensorRT are pre-installed and compatible.
- GPU drivers work out-of-the-box with NVIDIA Container Runtime.
- Replace 24.08-py3 with the latest tag matching your driver/CUDA.

✅ Performance Optimizations

- onnxruntime-gpu allows you to quickly validate ONNX inference before converting to TensorRT engines.
- polygraphy helps debug ONNX → TensorRT conversion issues.
- FP16 or INT8 precision can be enabled at engine build time for speed boost.

✅ Multilingual Support

- phonemizer supports IPA phoneme output for many languages.
- fastText or langdetect handles language detection.

✅ API Ready

- fastapi + uvicorn gives you a production-ready HTTP or WebSocket server for low-latency payloads.
