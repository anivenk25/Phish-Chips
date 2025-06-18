from typing import List, Dict
import torch
from inference import Extractor
from app.models import ChunkMetadata

def process(file_path: str,
            chunks: List[ChunkMetadata]) -> Dict:
    """
    Detects AI-generated or deepfake voices.
    """
    extractor = Extractor(
        encoder_model="damo/speech_personal_model",
        use_gpu=torch.cuda.is_available()
    )
    is_fake, score = extractor.detect_fake(file_path)
    return {
        "is_ai_voice": bool(is_fake),
        "ai_score": round(float(score), 4)
    }
