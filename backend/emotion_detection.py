"""
Per-segment emotion detection using wav2vec2-emotion-recognition.
Classifies audio into 8 emotions: angry, calm, disgust, fearful, happy, neutral, sad, surprised.
"""

import torch
import torchaudio
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
import torch.nn.functional as F

MODEL_NAME = "Dpngtm/wav2vec2-emotion-recognition"
LABELS = ["angry", "calm", "disgust", "fearful", "happy", "neutral", "sad", "surprised"]

# Deferred model loading for GPU memory carousel
_model = None
_processor = None
_device = None


def _load_model(force_gpu: bool = False):
    """Load model and processor on demand."""
    global _model, _processor, _device

    if _model is not None:
        return

    if force_gpu and not torch.cuda.is_available():
        raise RuntimeError(
            "GPU required for emotion detection but CUDA is not available."
        )

    if torch.cuda.is_available():
        _device = torch.device("cuda:0")
        torch.cuda.set_device(0)
        print(f"Emotion detection using GPU: {torch.cuda.get_device_name(0)}")
    else:
        _device = torch.device("cpu")
        print("Emotion detection falling back to CPU")

    _processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    _model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_NAME).to(_device)
    print(f"Emotion model loaded on: {next(_model.parameters()).device}")


def unload_model():
    """Explicitly unload model from GPU memory (called between carousel batches)."""
    global _model, _processor, _device
    if _model is not None:
        del _model
        _model = None
    if _processor is not None:
        del _processor
        _processor = None
    _device = None


def detect_emotion(audio_path: str, use_gpu: bool = True) -> dict:
    """
    Detect emotion in the given audio file.

    Returns:
        dict with 'top_emotion' (str) and 'scores' (dict mapping each emotion to its probability).
    """
    _load_model(force_gpu=use_gpu)

    waveform, sr = torchaudio.load(audio_path)

    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        if _device and _device.type == "cuda":
            resampler = resampler.to(_device)
            waveform = waveform.to(_device)
        waveform = resampler(waveform)

    waveform = waveform.mean(dim=0)  # mono

    # Processor requires numpy on CPU
    cpu_waveform = waveform.cpu().numpy() if waveform.is_cuda else waveform.numpy()
    inputs = _processor(
        cpu_waveform, sampling_rate=16000, return_tensors="pt", padding=True
    )

    if _device and _device.type == "cuda":
        inputs = {k: v.to(_device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = _model(**inputs).logits
        probs = F.softmax(logits, dim=-1)[0]

    scores = {label: float(probs[i]) for i, label in enumerate(LABELS)}
    top_emotion = max(scores, key=scores.get)

    return {"top_emotion": top_emotion, "scores": scores}


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "test.wav"
    result = detect_emotion(path)
    print(f"Top Emotion: {result['top_emotion'].upper()}")
    for label, prob in sorted(
        result["scores"].items(), key=lambda x: x[1], reverse=True
    ):
        print(f"  {label:<10}: {prob:.4f}")
