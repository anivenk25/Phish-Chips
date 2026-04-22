import torch
import torchaudio
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

# FORCE GPU-ONLY MODE FOR AZURE - with intelligent fallback
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
    torch.cuda.set_device(0)
    print(f"🕵️ AI Voice Detection using GPU: {torch.cuda.get_device_name(0)}")
else:
    DEVICE = torch.device("cpu")
    print("⚠️ AI Voice Detection falling back to CPU")

model_name = "as1605/Deepfake-audio-detection-V2"

# Load the model & extractor - Use optimal device
model = AutoModelForAudioClassification.from_pretrained(model_name, weights_only=True).to(DEVICE)
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

# Verify model device
print(f"🕵️ AI Voice Detection model loaded on: {next(model.parameters()).device}")

def load_audio(path, target_sr=16000):
    """Load audio and use optimal device for processing"""
    wav, sr = torchaudio.load(path)
    
    # Move to device for processing if GPU available
    if torch.cuda.is_available():
        wav = wav.to(DEVICE)
        
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr).to(DEVICE)
            wav = resampler(wav)
        
        # Return CPU numpy for feature extractor
        return wav.squeeze().cpu().numpy()
    else:
        # CPU processing
        if sr != target_sr:
            wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
        
        return wav.squeeze().numpy()

def detect_deepfake(path, threshold=1.5, force_gpu=True):
    """Detect AI-generated voice using GPU when available"""
    device = DEVICE if torch.cuda.is_available() else torch.device("cpu")
    
    audio = load_audio(path)
    inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    
    # Move inputs to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
    
    # Assumes index=1 is 'fake'
    fake_prob = probs[1].item()
    return fake_prob > threshold, fake_prob

def detect_deepfake(path, threshold=1.5, force_gpu=True):
    """Detect AI-generated voice using GPU ONLY"""
    if not torch.cuda.is_available():
        raise RuntimeError("❌ GPU required for AI voice detection!")
    
    audio = load_audio(path)
    inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    
    # Force inputs to GPU
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
    
    # Assumes index=1 is 'fake'
    fake_prob = probs[1].item()
    return fake_prob > threshold, fake_prob

# Example usage when run as a script
if __name__ == "__main__":
    path = "/home/r12/Downloads/1.wav"
    is_fake, score = detect_deepfake(path)
    print(f"Fake probability: {score:.1%} — Classified as {'FAKE' if is_fake else 'REAL'}")
