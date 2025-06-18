import torch
import torchaudio
import librosa
import soundfile as sf
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModelForAudioClassification

# --- CONFIG ---
AUDIO_PATH = "vishkill_basic.py"  
MODEL_NAME = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"

# --- STEP 1: Convert to WAV Mono 16k ---
def convert_to_wav_mono_16k(input_path):
    y, sr = librosa.load(input_path, sr=16000, mono=True)
    output_path = "converted_audio.wav"
    sf.write(output_path, y, 16000)
    return output_path

# --- STEP 2: Load Model & Processor ---
processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForAudioClassification.from_pretrained(MODEL_NAME)

# --- STEP 3: Load + Preprocess Audio ---
wav_path = convert_to_wav_mono_16k(AUDIO_PATH)
waveform, sr = torchaudio.load(wav_path)
inputs = processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt")

# --- STEP 4: Predict ---
with torch.no_grad():
    logits = model(**inputs).logits
    probs = F.softmax(logits, dim=1)[0]
    label_probs = {model.config.id2label[i]: float(probs[i]) for i in range(len(probs))}
    top_emotion = max(label_probs, key=label_probs.get)

# --- STEP 5: Output in Clean Format ---
print(f"\n🎧 Top Emotion: **{top_emotion.upper()}**\n")
print("📊 Emotion Probabilities:")
for label, prob in sorted(label_probs.items(), key=lambda x: x[1], reverse=True):
    print(f"  {label:<10} : {prob:.4f}")
