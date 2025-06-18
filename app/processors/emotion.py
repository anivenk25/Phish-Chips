from typing import List, Dict
import torch
import torchaudio
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModelForAudioClassification
from app.models import ChunkMetadata

MODEL_NAME = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"

def process(file_path: str,
            chunks: List[ChunkMetadata]) -> List[Dict]:
    """
    Classifies emotions in audio segments using wav2vec2 emotion model.
    """
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForAudioClassification.from_pretrained(MODEL_NAME)
    waveform, sr = torchaudio.load(file_path)
    inputs = processor(waveform.squeeze(), sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = F.softmax(logits, dim=1)[0]
    label_probs = {model.config.id2label[i]: float(probs[i]) for i in range(len(probs))}
    top_emotion = max(label_probs, key=label_probs.get)
    return [{
        "top_emotion": top_emotion,
        "emotion_probs": label_probs
    }]
