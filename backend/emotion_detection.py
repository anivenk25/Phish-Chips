import torch
import torchaudio
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
import torch.nn.functional as F

# FORCE GPU-ONLY MODE FOR AZURE - with fallback
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
    torch.cuda.set_device(0)
    print(f"🎭 Emotion detection using GPU: {torch.cuda.get_device_name(0)}")
else:
    DEVICE = torch.device("cpu")
    print("⚠️ Emotion detection falling back to CPU")

# Configuration
MODEL_NAME = "Dpngtm/wav2vec2-emotion-recognition"

# Load processor and model once at import - FORCE GPU
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_NAME).to(DEVICE)

# Verify model is on GPU
print(f"🎭 Emotion model loaded on: {next(model.parameters()).device}")

def detect_emotion(audio_path: str, use_gpu: bool = True) -> dict:
    """
    Detects emotion in the given audio file using GPU when available.
    Returns a dict with 'top_emotion' and 'scores' mapping each emotion to its probability.
    """
    # Use GPU when available, otherwise fallback to current device
    device = DEVICE if torch.cuda.is_available() else torch.device("cpu")
    
    waveform, sr = torchaudio.load(audio_path)
    
    # Move waveform to device if GPU available
    if torch.cuda.is_available():
        waveform = waveform.to(device)
        
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000).to(device)
            waveform = resampler(waveform)
        
        waveform = waveform.mean(dim=0)  # mono
        
        # Move to CPU for processor, then back to device
        inputs = processor(waveform.cpu().numpy(), sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
    else:
        # CPU processing
        if sr != 16000:
            waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
        
        waveform = waveform.mean(dim=0)  # mono
        inputs = processor(waveform.numpy(), sampling_rate=16000, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=-1)[0]
    
    labels = ["angry", "calm", "disgust", "fearful", "happy", "neutral", "sad", "surprised"]
    scores = {label: float(probs[i]) for i, label in enumerate(labels)}
    top_emotion = max(scores, key=scores.get)
    
    return {"top_emotion": top_emotion, "scores": scores}

if __name__ == "__main__":
    # Example usage for local testing
    import librosa
    import soundfile as sf

    input_path = "/home/r12/Downloads/WhatsApp Ptt 2025-06-19 at 11.38.48 PM.ogg"
    # Convert to WAV mono 16k
    y, sr = librosa.load(input_path, sr=16000, mono=True)
    wav_path = "converted_audio.wav"
    sf.write(wav_path, y, 16000)
    result = detect_emotion(wav_path)
    print(f"Top Emotion: {result['top_emotion'].upper()}")
    print("All Emotion Probabilities:")
    for label, prob in sorted(result["scores"].items(), key=lambda x: x[1], reverse=True):
        print(f"  {label:<10}: {prob:.4f}")
