import os
from typing import List, Dict
from pyannote.audio import Pipeline
from app.models import ChunkMetadata
import torch


def process(file_path: str, chunks: List[ChunkMetadata]) -> List[Dict]:
    """
    Identifies speaker segments using pyannote.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Hugging Face token from environment
    hf_token = (
        os.getenv("HUGGINGFACE_TOKEN")
        or os.getenv("HF_HUB_TOKEN")
        or os.getenv("HUGGINGFACE_HUB_TOKEN")
    )
    use_token = hf_token if hf_token else False

    # Load speaker diarization pipeline
    pipeline_raw = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=use_token
    )

    if pipeline_raw is None:
        raise RuntimeError(
            "Failed to load pyannote speaker-diarization pipeline. "
            "Please set the HUGGINGFACE_TOKEN environment variable with "
            "a valid access token and accept the model's terms at "
            "https://hf.co/pyannote/speaker-diarization-3.1"
        )

    # Move pipeline to device, fallback to CPU on OOM
    try:
        pipeline = pipeline_raw.to(device)
    except RuntimeError as e:
        if "out of memory" in str(e).lower() and device.type == "cuda":
            pipeline = pipeline_raw.to(torch.device("cpu"))
        else:
            raise

    diarization = pipeline(file_path)
    segments: List[Dict] = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "speaker": speaker,
            "start": round(turn.start, 2),
            "end": round(turn.end, 2),
            "duration": round(turn.duration, 2)
        })
    return segments

