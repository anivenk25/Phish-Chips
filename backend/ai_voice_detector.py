"""
Deepfake / AI-cloned voice detection using as1605/Deepfake-audio-detection-V2.
Outputs a boolean (is_fake) and a float confidence score (0.0 - 1.0).
"""

import torch
import torchaudio
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

MODEL_NAME = "as1605/Deepfake-audio-detection-V2"

# Deferred model loading -- models are loaded on first call to detect_deepfake()
# so that they don't consume GPU memory at import time (supports GPU memory carousel).
_model = None
_feature_extractor = None
_device = None


def _load_model(force_gpu: bool = False):
    """Load model and feature extractor on demand."""
    global _model, _feature_extractor, _device

    if _model is not None:
        return

    if force_gpu and not torch.cuda.is_available():
        raise RuntimeError(
            "GPU required for AI voice detection but CUDA is not available."
        )

    if torch.cuda.is_available():
        _device = torch.device("cuda:0")
        torch.cuda.set_device(0)
        print(f"AI Voice Detection using GPU: {torch.cuda.get_device_name(0)}")
    else:
        _device = torch.device("cpu")
        print("AI Voice Detection falling back to CPU")

    _model = AutoModelForAudioClassification.from_pretrained(MODEL_NAME).to(_device)
    _feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
    print(f"AI Voice Detection model loaded on: {next(_model.parameters()).device}")


def unload_model():
    """Explicitly unload model from GPU memory (called between carousel batches)."""
    global _model, _feature_extractor, _device
    if _model is not None:
        del _model
        _model = None
    if _feature_extractor is not None:
        del _feature_extractor
        _feature_extractor = None
    _device = None


def _load_audio(path: str, target_sr: int = 16000):
    """Load and resample audio to target sample rate, return numpy array."""
    wav, sr = torchaudio.load(path)

    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        wav = resampler(wav)

    return wav.squeeze().numpy()


def detect_deepfake(path: str, threshold: float = 0.5, force_gpu: bool = False):
    """
    Detect whether an audio file contains AI-generated / deepfake speech.

    Args:
        path: Path to the audio file.
        threshold: Probability threshold above which the voice is classified as fake.
                   Default 0.5 (majority probability).
        force_gpu: If True, raise an error when no GPU is available.

    Returns:
        (is_fake: bool, fake_probability: float)
    """
    _load_model(force_gpu=force_gpu)

    audio = _load_audio(path)
    inputs = _feature_extractor(
        audio, sampling_rate=16000, return_tensors="pt", padding=True
    )

    # Move inputs to the same device as the model
    inputs = {k: v.to(_device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = _model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]

    # Index 1 is the 'fake' class
    fake_prob = probs[1].item()
    return fake_prob > threshold, fake_prob


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "test.wav"
    is_fake, score = detect_deepfake(path)
    print(
        f"Fake probability: {score:.1%} -- Classified as {'FAKE' if is_fake else 'REAL'}"
    )
