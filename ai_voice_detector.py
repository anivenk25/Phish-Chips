import torch
from inference import Extractor
import sys

audio_path = "/home/r12/Downloads/lol.ogg"  

extractor = Extractor(encoder_model='damo/speech_personal_model', use_gpu=torch.cuda.is_available())


is_fake, score = extractor.detect_fake(audio_path)

print(f"\nFile: {audio_path}")
print(f"Fake Voice Probability Score: {score:.4f}")

if is_fake:
    print("Detected as AI-generated / Deepfake voice.")
else:
    print("Detected as real human voice.")