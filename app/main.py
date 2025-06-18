from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from concurrent.futures import ThreadPoolExecutor
import os
import tempfile
import logging

import librosa
import soundfile as sf

from app.config import settings
from app.models import AnalysisResponse
from app.utils import secure_delete, split_audio_into_chunks
from app.processors import ambience, diarization, ai_voice, emotion, transcription, scam_analysis

logger = logging.getLogger("audio_analysis")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Audio Analysis Service")

@app.on_event("startup")
def startup_event():
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    logger.info(f"Upload directory: {settings.UPLOAD_DIR}")

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_audio(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    filename = file.filename
    if not filename.lower().endswith((".wav", ".mp3", ".flac", ".m4a", ".ogg")):
        raise HTTPException(status_code=400, detail="Unsupported file type.")
    logger.info(f"Received file: {filename}")
    orig_fd, orig_path = tempfile.mkstemp(suffix=os.path.splitext(filename)[1])
    os.close(orig_fd)
    content = await file.read()
    with open(orig_path, "wb") as f:
        f.write(content)
    try:
        y, sr = librosa.load(orig_path, sr=16000, mono=True)
    except Exception:
        secure_delete(orig_path)
        raise HTTPException(status_code=400, detail="Failed to process audio file.")
    wav_fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(wav_fd)
    sf.write(wav_path, y, 16000)
    info = sf.info(wav_path)
    duration = info.duration
    logger.info(f"Converted audio: duration={duration:.2f}s, sample_rate={info.samplerate}")
    metadata = split_audio_into_chunks(wav_path, settings.CHUNK_DURATION_SECONDS, settings.CHUNK_OVERLAP_SECONDS)
    logger.info(f"Audio split into {len(metadata)} chunks")
    transcription_res = transcription.process(wav_path, metadata)
    with ThreadPoolExecutor(max_workers=settings.MAX_WORKERS) as executor:
        future_ambience = executor.submit(ambience.process, wav_path)
        future_diarization = executor.submit(diarization.process, wav_path, metadata)
        future_ai_voice = executor.submit(ai_voice.process, wav_path, metadata)
        future_emotion = executor.submit(emotion.process, wav_path, metadata)
        future_scam = executor.submit(scam_analysis.process, transcription_res)
        ambience_res = future_ambience.result()
        diarization_res = future_diarization.result()
        ai_voice_res = future_ai_voice.result()
        emotion_res = future_emotion.result()
        scam_res = future_scam.result()
    background_tasks.add_task(secure_delete, orig_path)
    background_tasks.add_task(secure_delete, wav_path)
    return AnalysisResponse(
        ambience=ambience_res,
        diarization=diarization_res,
        ai_voice=ai_voice_res,
        emotion=emotion_res,
        transcription=transcription_res,
        scam_analysis=scam_res,
        chunks=metadata
    )
