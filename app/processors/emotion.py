from typing import List, Dict, Optional
import torch
import torchaudio
import torch.nn.functional as F
import librosa
from transformers import AutoProcessor, AutoModelForAudioClassification
from app.models import ChunkMetadata

# Configuration
MODEL_NAME = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
SAMPLE_RATE = 16000

# Lazy-load model and processor on first use
_processor: Optional[AutoProcessor] = None
_model: Optional[AutoModelForAudioClassification] = None

def process(file_path: str,
            chunks: List[ChunkMetadata]) -> List[Dict]:
    """
    Classifies emotions in an audio file using wav2vec2 emotion model.
    Converts input to mono WAV at 16 kHz if needed.
    """
    # Load audio; torchaudio returns shape (channels, time)
    waveform, sr = torchaudio.load(file_path)
    # Convert to single-channel 16 kHz if not already
    if waveform.shape[0] != 1 or sr != SAMPLE_RATE:
        y, _ = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
        waveform = torch.from_numpy(y).unsqueeze(0)
        sr = SAMPLE_RATE

    # Initialize processor and model if not already loaded
    global _processor, _model
    if _processor is None or _model is None:
        _processor = AutoProcessor.from_pretrained(MODEL_NAME)
        _model = AutoModelForAudioClassification.from_pretrained(MODEL_NAME)
    # Prepare inputs
    inputs = _processor(waveform.squeeze(), sampling_rate=sr, return_tensors="pt")
    # Inference
    with torch.no_grad():
        logits = _model(**inputs).logits
    probs = F.softmax(logits, dim=1)[0]
    # Map to labels
    label_probs = { _model.config.id2label[i]: float(probs[i])
                    for i in range(probs.shape[0]) }
    top_emotion = max(label_probs, key=label_probs.get)
    return [{
        "top_emotion": top_emotion,
        "emotion_probs": label_probs
    }]
