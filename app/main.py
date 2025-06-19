from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from concurrent.futures import ThreadPoolExecutor
import os
import tempfile
import logging

import librosa
import soundfile as sf

from app.config import settings
from app.models import AnalysisResponse
from app.utils import secure_delete, split_audio_into_chunks, load_audio_chunk
from app.processors import ambience, diarization, ai_voice, emotion, transcription, scam_analysis

logger = logging.getLogger("audio_analysis")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Audio Analysis Service")

@app.on_event("startup")
def startup_event():
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    logger.info(f"Upload directory: {settings.UPLOAD_DIR}")
    
class AnalysisService:
    """
    Orchestrates analysis tasks with resource-aware sequencing.
    """
    def __init__(self, wav_path, metadata):
        self.wav_path = wav_path
        self.metadata = metadata

    def run(self):
        # Fallback to whole-file processing if no chunk metadata (e.g., unit tests)
        if not self.metadata:
            # Sequential GPU-intensive tasks
            diarization_res = diarization.process(self.wav_path, self.metadata)
            ai_voice_res = ai_voice.process(self.wav_path, self.metadata)
            # CPU-bound tasks
            ambience_res = ambience.process(self.wav_path)
            emotion_res = emotion.process(self.wav_path, self.metadata)
            transcription_res = transcription.process(self.wav_path, self.metadata)
            # Scam analysis depends on transcription, AI voice, and emotion
            scam_res = scam_analysis.process(
                transcription_res,
                ai_voice_res,
                emotion_res
            )
            return {
                'ambience': ambience_res,
                'diarization': diarization_res,
                'ai_voice': ai_voice_res,
                'emotion': emotion_res,
                'transcription': transcription_res,
                'scam_analysis': scam_res
            }
        # Chunk-level processing: run each analysis module on overlapping chunks, then aggregate.
        diarization_segments = []
        all_transcripts = []
        transcription_segments = []
        ai_results = []
        ambience_scores = []
        ambience_tags = set()
        emotion_probs_list = []

        # Unique ID counter for transcription segments
        seg_id_counter = 0

        # Process each chunk
        for chunk in self.metadata:
            # Extract chunk audio and write to temp WAV file
            data, sr = load_audio_chunk(self.wav_path, chunk.start_time, chunk.end_time)
            tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav")
            os.close(tmp_fd)
            sf.write(tmp_path, data, sr)

            try:
                # Diarization on chunk
                diar_res = diarization.process(tmp_path, [chunk])
                for seg in diar_res:
                    seg['start'] = round(seg['start'] + chunk.start_time, 2)
                    seg['end'] = round(seg['end'] + chunk.start_time, 2)
                    diarization_segments.append(seg)

                # AI voice detection on chunk
                ai_res = ai_voice.process(tmp_path, [chunk])
                ai_results.append(ai_res)

                # Ambience detection on chunk
                amb_res = ambience.process(tmp_path)
                ambience_scores.append(amb_res.get('composite_score', 0.0))
                ambience_tags.update(amb_res.get('detected_tags', []))

                # Emotion detection on chunk
                emo_list = emotion.process(tmp_path, [chunk])
                if emo_list:
                    emotion_probs_list.append(emo_list[0].get('emotion_probs', {}))

                # Transcription on chunk
                trans_res = transcription.process(tmp_path, [chunk])
                chunk_text = trans_res.get('transcript', '').strip()
                all_transcripts.append(chunk_text)
                for seg in trans_res.get('segments', []):
                    seg['id'] = seg_id_counter
                    seg_id_counter += 1
                    seg['start'] = round(seg['start'] + chunk.start_time, 2)
                    seg['end'] = round(seg['end'] + chunk.start_time, 2)
                    transcription_segments.append(seg)
            finally:
                # Clean up temporary chunk file
                secure_delete(tmp_path)

        # Aggregate diarization segments
        diarization_res = diarization_segments

        # Aggregate AI voice results
        ai_scores = [r.get('ai_score', 0.0) for r in ai_results]
        is_ai = any(r.get('is_ai_voice', False) for r in ai_results)
        ai_voice_res = {
            'is_ai_voice': bool(is_ai),
            'ai_score': round(max(ai_scores) if ai_scores else 0.0, 4)
        }

        # Aggregate ambience results
        avg_score = sum(ambience_scores) / len(ambience_scores) if ambience_scores else 0.0
        ambience_res = {
            'is_office': avg_score > 0.6,
            'confidence': float(avg_score),
            'detected_tags': list(ambience_tags),
            'composite_score': float(avg_score)
        }

        # Aggregate emotion probabilities
        emotion_res = []
        if emotion_probs_list:
            merged = {}
            for probs in emotion_probs_list:
                for label, p in probs.items():
                    merged[label] = merged.get(label, 0.0) + p
            total = len(emotion_probs_list)
            merged = {label: float(p / total) for label, p in merged.items()}
            top_emotion = max(merged, key=merged.get)
            emotion_res = [{'top_emotion': top_emotion, 'emotion_probs': merged}]

        # Aggregate transcription
        full_transcript = ' '.join([t for t in all_transcripts if t])
        transcription_res = {
            'transcript': full_transcript,
            'segments': transcription_segments
        }

        # Scam analysis based on aggregated results
        scam_res = scam_analysis.process(transcription_res, ai_voice_res, emotion_res)

        return {
            'ambience': ambience_res,
            'diarization': diarization_res,
            'ai_voice': ai_voice_res,
            'emotion': emotion_res,
            'transcription': transcription_res,
            'scam_analysis': scam_res
        }

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
    # Run analysis service for resource-aware execution
    service = AnalysisService(wav_path, metadata)
    results = service.run()
    ambience_res = results["ambience"]
    diarization_res = results["diarization"]
    ai_voice_res = results["ai_voice"]
    emotion_res = results["emotion"]
    transcription_res = results["transcription"]
    scam_res = results["scam_analysis"]
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
