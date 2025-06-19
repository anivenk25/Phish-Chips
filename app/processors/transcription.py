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
    # Use translation task to translate speech to English while transcribing
    result = model.transcribe(file_path, task="translate")
    # Extract and clean transcript text
    transcript = result.get("text", "").strip()
    segments = result.get("segments", [])
    return {
        "transcript": transcript,
        "segments": segments
    }
