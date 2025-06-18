import torch
import torchaudio
import torch.nn.functional as F
import subprocess
import json
import whisper
import librosa
import soundfile as sf
import concurrent.futures
from inference import Extractor
from pyannote.audio import Pipeline
from transformers import AutoProcessor, AutoModelForAudioClassification

# --- PATH CONFIG ---
AUDIO_PATH = "/home/r12/Downloads/lol.ogg"
CONVERTED_WAV = "converted_audio.wav"

# --- STEP 1: Convert to WAV Mono 16k ---
def convert_to_wav_mono_16k(input_path):
    y, sr = librosa.load(input_path, sr=16000, mono=True)
    sf.write(CONVERTED_WAV, y, 16000)
    return CONVERTED_WAV

# --- STEP 2: Speaker Diarization ---
def diarize_audio(audio_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=False).to(device)
    diarization = diarization_pipeline(audio_path)
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({"speaker": speaker, "start": round(turn.start, 2), "end": round(turn.end, 2), "duration": round(turn.duration, 2)})
    return segments

# --- STEP 3: AI Voice Detection ---
def detect_ai_voice(audio_path):
    extractor = Extractor(encoder_model='damo/speech_personal_model', use_gpu=torch.cuda.is_available())
    is_fake, score = extractor.detect_fake(audio_path)
    result = {
        "is_ai_voice": is_fake,
        "ai_score": round(score, 4)
    }
    return result

# --- STEP 4: Emotion Detection ---
def detect_emotion(audio_path):
    MODEL_NAME = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForAudioClassification.from_pretrained(MODEL_NAME)
    waveform, sr = torchaudio.load(audio_path)
    inputs = processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=1)[0]
        label_probs = {model.config.id2label[i]: float(probs[i]) for i in range(len(probs))}
        top_emotion = max(label_probs, key=label_probs.get)
    return {
        "top_emotion": top_emotion,
        "emotion_probs": label_probs
    }

# --- STEP 5: Transcription + Scam Detection ---
def transcribe_and_analyze(audio_path, ai_result, emotion_result):
    model = whisper.load_model("medium")
    result = model.transcribe(audio_path, task="translate")
    transcript = result["text"].strip()

    prompt = f"""
Analyze this call for scam indicators. Include:
- Emotion: {emotion_result['top_emotion']}
- AI Voice Detected: {ai_result['is_ai_voice']} (score: {ai_result['ai_score']})
- Focus on scam types: phishing, tech support, IRS, fake prizes, impersonation
- Red flags: urgency, requests for money/personal info

Respond in this JSON format:
{{
    "is_scam": boolean,
    "confidence": 0-100,
    "red_flags": ["..."],
    "target": ["money"|"personal_info"|"credentials"|"none"],
    "analysis": "..."
}}

Transcript: "{transcript[:2000]}"
"""

    cmd = ["ollama", "run", "hermes3:8b", prompt]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        json_str = result.stdout[result.stdout.find('{'):result.stdout.rfind('}')+1]
        analysis = json.loads(json_str)
        print("\n=== SCAM ANALYSIS ===")
        print(json.dumps(analysis, indent=4))
    except json.JSONDecodeError:
        print("Failed to parse analysis results")
        print(result.stdout)

# --- MAIN ---
if __name__ == "__main__":
    wav_path = convert_to_wav_mono_16k(AUDIO_PATH)

    print("[1] Starting speaker diarization...")
    segments = diarize_audio(wav_path)
    print("\n[SPEAKER SEGMENTS]")
    for seg in segments:
        print(f"{seg['speaker']}: {seg['start']}s → {seg['end']}s")

    print("[2] Running voice/AI/emotion detection in parallel...")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_ai = executor.submit(detect_ai_voice, wav_path)
        future_emotion = executor.submit(detect_emotion, wav_path)

        ai_result = future_ai.result()
        emotion_result = future_emotion.result()

    print("\n[AI VOICE DETECTION]")
    print(ai_result)

    print("\n[EMOTION DETECTION]")
    print(f"Top Emotion: {emotion_result['top_emotion']}")

    print("[3] Running transcription + scam analysis")
    transcribe_and_analyze(wav_path, ai_result, emotion_result)
