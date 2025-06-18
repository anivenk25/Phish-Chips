from typing import List, Dict, Any
from pydantic import BaseModel

class ChunkMetadata(BaseModel):
    index: int
    start_time: float
    end_time: float

class AnalysisResponse(BaseModel):
    ambience: Dict[str, Any]
    diarization: List[Dict[str, Any]]
    ai_voice: Dict[str, Any]
    emotion: List[Dict[str, Any]]
    transcription: Dict[str, Any]
    scam_analysis: Dict[str, Any]
    chunks: List[ChunkMetadata]
