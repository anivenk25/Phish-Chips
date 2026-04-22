from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import os
import time
import json
from pathlib import Path
import asyncio

from main import analyze_audio_file

app = FastAPI(title="VoIP Scam Detection API")

RECORDINGS_DIR = os.getenv("RECORDINGS_DIR", "/app/recordings")
RESULTS_DIR = os.getenv("RESULTS_DIR", "/app/results")

@app.get("/voip-status")
async def voip_status():
    return {"status": "running", "recordings_dir": RECORDINGS_DIR, "results_dir": RESULTS_DIR}

@app.post("/analyze-audio")
async def analyze_audio_endpoint(file: UploadFile = File(...)):
    """
    Accepts a raw audio file upload, saves it to RECORDINGS_DIR, analyzes it, and returns the result.
    """
    # Ensure directories exist
    os.makedirs(RECORDINGS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Save uploaded file
    timestamp = int(time.time())
    filename = f"{timestamp}_{file.filename}"
    file_path = os.path.join(RECORDINGS_DIR, filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Analyze
    result = await analyze_audio_file(file_path)
    return JSONResponse(result)

@app.post("/force-analyze/{filename}")
async def force_analyze(filename: str):
    """
    Trigger analysis for an existing file in RECORDINGS_DIR by filename.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    file_path = os.path.join(RECORDINGS_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    result = await analyze_audio_file(file_path)
    return JSONResponse(result)

@app.get("/analysis-results")
async def analysis_results():
    """
    Return all JSON result files in RESULTS_DIR.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = []
    for fname in sorted(os.listdir(RESULTS_DIR)):
        if fname.lower().endswith(".json"):
            path = os.path.join(RESULTS_DIR, fname)
            try:
                with open(path) as f:
                    data = json.load(f)
                results.append(data)
            except:
                continue
    return results

# Note: Run with `uvicorn api:app --host 0.0.0.0 --port 8000`