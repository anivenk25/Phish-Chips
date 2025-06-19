from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from concurrent.futures import ThreadPoolExecutor
import asyncio
import os
import tempfile
import logging

import librosa
import soundfile as sf

from app.config import settings
from app.models import AnalysisResponse, ChunkMetadata
from app.utils import secure_delete, split_audio_into_chunks, load_audio_chunk
from app.processors import ambience, diarization, ai_voice, emotion, transcription, scam_analysis

logger = logging.getLogger("audio_analysis")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Audio Analysis Service")
# Enable CORS for the web UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
def index():
    html_content = """<!DOCTYPE html>
<html>
<head>
  <title>Audio Analysis Service - MVP</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 40px; }
    input[type="file"] { margin-bottom: 10px; }
    pre { background: #f4f4f4; padding: 10px; border-radius: 5px; }
  </style>
</head>
<body>
  <h1>Audio Analysis Service - MVP</h1>
  <p>Upload an audio file and receive analysis results.</p>
  <input type="file" id="fileInput" accept=".wav,.mp3,.flac,.m4a,.ogg"/>
  <button onclick="upload()">Analyze</button>
  <h2>Results:</h2>
  <pre id="results"></pre>
  <script>
  async function upload() {
     const fileInput = document.getElementById('fileInput');
     if (!fileInput.files.length) {
        alert('Please select a file.');
        return;
     }
     const file = fileInput.files[0];
     const formData = new FormData();
     formData.append('file', file);
     const res = await fetch('/analyze', {
        method: 'POST',
        body: formData
     });
     const resultsEl = document.getElementById('results');
     if (!res.ok) {
        const err = await res.json();
        resultsEl.textContent = 'Error: ' + (err.detail || JSON.stringify(err));
        return;
     }
     const data = await res.json();
     resultsEl.textContent = JSON.stringify(data, null, 2);
  }
  </script>
</body>
</html>"""
    return html_content

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
        # Chunk-level processing: run each analysis module on overlapping chunks in parallel, then aggregate.
        def _process_chunk(chunk):
            data, sr = load_audio_chunk(self.wav_path, chunk.start_time, chunk.end_time)
            tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav")
            os.close(tmp_fd)
            sf.write(tmp_path, data, sr)
            try:
                diar_res = diarization.process(tmp_path, [chunk])
                ai_res = ai_voice.process(tmp_path, [chunk])
                amb_res = ambience.process(tmp_path)
                emo_list = emotion.process(tmp_path, [chunk])
                emo_probs = emo_list[0].get('emotion_probs', {}) if emo_list else {}
                trans_res = transcription.process(tmp_path, [chunk])
                transcript_text = trans_res.get('transcript', '').strip()
                trans_segs = trans_res.get('segments', [])
            finally:
                secure_delete(tmp_path)
            return {
                'index': chunk.index,
                'start': chunk.start_time,
                'diar': diar_res,
                'ai': ai_res,
                'amb': amb_res,
                'emo': emo_probs,
                'transcript': transcript_text,
                'trans_segs': trans_segs
            }

        with ThreadPoolExecutor(max_workers=settings.MAX_WORKERS) as executor:
            chunk_results = list(executor.map(_process_chunk, self.metadata))

        chunk_results.sort(key=lambda x: x['index'])
        diarization_segments = []
        ai_results = []
        ambience_scores = []
        ambience_tags = set()
        emotion_probs_list = []
        all_transcripts = []
        transcription_segments = []
        seg_id_counter = 0

        # Aggregate per-chunk results
        for r in chunk_results:
            offset = r['start']
            for seg in r['diar']:
                seg_copy = dict(seg)
                seg_copy['start'] = round(seg_copy['start'] + offset, 2)
                seg_copy['end'] = round(seg_copy['end'] + offset, 2)
                diarization_segments.append(seg_copy)
            ai_results.append(r['ai'])
            ambience_scores.append(r['amb'].get('composite_score', 0.0))
            ambience_tags.update(r['amb'].get('detected_tags', []))
            if r['emo']:
                emotion_probs_list.append(r['emo'])
            all_transcripts.append(r['transcript'])
            for seg in r['trans_segs']:
                seg_copy = dict(seg)
                seg_copy['id'] = seg_id_counter
                seg_id_counter += 1
                seg_copy['start'] = round(seg_copy['start'] + offset, 2)
                seg_copy['end'] = round(seg_copy['end'] + offset, 2)
                transcription_segments.append(seg_copy)

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
        full_transcript = ' '.join(t for t in all_transcripts if t)
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
    # Ensure upload directory exists (may not have run startup in TestClient)
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    orig_fd, orig_path = tempfile.mkstemp(suffix=os.path.splitext(filename)[1], dir=settings.UPLOAD_DIR)
    os.close(orig_fd)
    content = await file.read()
    with open(orig_path, "wb") as f:
        f.write(content)
    try:
        # Load at original sample rate to preserve full bandwidth
        y, sr = librosa.load(orig_path, sr=None, mono=True)
    except Exception:
        secure_delete(orig_path)
        raise HTTPException(status_code=400, detail="Failed to process audio file.")
    wav_fd, wav_path = tempfile.mkstemp(suffix=".wav", dir=settings.UPLOAD_DIR)
    os.close(wav_fd)
    sf.write(wav_path, y, sr)
    info = sf.info(wav_path)
    duration = info.duration
    logger.info(f"Converted audio: duration={duration:.2f}s, sample_rate={info.samplerate}")
    metadata = split_audio_into_chunks(wav_path, settings.CHUNK_DURATION_SECONDS, settings.CHUNK_OVERLAP_SECONDS)
    logger.info(f"Audio split into {len(metadata)} chunks")
    # Run analysis service in a thread to avoid blocking the event loop
    service = AnalysisService(wav_path, metadata)
    loop = asyncio.get_running_loop()
    try:
        results = await loop.run_in_executor(None, service.run)
    finally:
        # Always schedule cleanup of temp files
        background_tasks.add_task(secure_delete, orig_path)
        background_tasks.add_task(secure_delete, wav_path)
    ambience_res = results["ambience"]
    diarization_res = results["diarization"]
    ai_voice_res = results["ai_voice"]
    emotion_res = results["emotion"]
    transcription_res = results["transcription"]
    scam_res = results["scam_analysis"]
    return AnalysisResponse(
        ambience=ambience_res,
        diarization=diarization_res,
        ai_voice=ai_voice_res,
        emotion=emotion_res,
        transcription=transcription_res,
        scam_analysis=scam_res,
        chunks=metadata
    )
