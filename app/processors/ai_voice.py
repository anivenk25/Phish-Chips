from typing import List, Dict
import torch
from inference import Extractor
from app.models import ChunkMetadata

def process(file_path: str,
            chunks: List[ChunkMetadata]) -> Dict:
    """
    Detects AI-generated or deepfake voices.
    """
    use_gpu = torch.cuda.is_available()
    try:
        extractor = Extractor(
            encoder_model="damo/speech_personal_model",
            use_gpu=use_gpu
        )
        is_fake, score = extractor.detect_fake(file_path)
    except RuntimeError as e:
        # Fallback to CPU if GPU out of memory
        if "out of memory" in str(e).lower() and use_gpu:
            extractor = Extractor(
                encoder_model="damo/speech_personal_model",
                use_gpu=False
            )
            is_fake, score = extractor.detect_fake(file_path)
        else:
            raise
    return {
        "is_ai_voice": bool(is_fake),
        "ai_score": round(float(score), 4)
    }
