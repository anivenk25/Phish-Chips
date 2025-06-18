import torch
from pyannote.audio import Pipeline
from typing import List, Dict
from app.models import ChunkMetadata

def process(file_path: str,
            chunks: List[ChunkMetadata]) -> List[Dict]:
    """
    Identifies speaker segments using pyannote.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", use_auth_token=False
    ).to(device)
    diarization = pipeline(file_path)
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "speaker": speaker,
            "start": round(turn.start, 2),
            "end": round(turn.end, 2),
            "duration": round(turn.duration, 2)
        })
    return segments
