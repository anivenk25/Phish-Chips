from typing import List, Dict, Any
import whisper
from app.models import ChunkMetadata

def process(file_path: str,
            chunks: List[ChunkMetadata]) -> Dict[str, Any]:
    """
    Transcribes audio using Whisper and returns full transcript and segments.
    """
    # Load Whisper model on CPU to avoid GPU contention
    model = whisper.load_model("medium", device="cpu")
    result = model.transcribe(file_path)
    transcript = result.get("text", "")
    segments = result.get("segments", [])
    return {
        "transcript": transcript,
        "segments": segments
    }
