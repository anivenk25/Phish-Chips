import os
from typing import List, Tuple
import soundfile as sf
from app.models import ChunkMetadata

def split_audio_into_chunks(file_path: str,
                             chunk_duration: float,
                             overlap: float) -> List[ChunkMetadata]:
    info = sf.info(file_path)
    total_duration = info.duration
    step = chunk_duration - overlap
    chunks = []
    start = 0.0
    index = 0
    while start < total_duration:
        end = min(start + chunk_duration, total_duration)
        chunks.append(ChunkMetadata(index=index, start_time=start, end_time=end))
        start += step
        index += 1
    return chunks


def load_audio_chunk(file_path: str,
                     start: float,
                     end: float) -> Tuple:
    info = sf.info(file_path)
    sr = info.samplerate
    start_sample = int(start * sr)
    end_sample = int(end * sr)
    data, _ = sf.read(file_path, start=start_sample, stop=end_sample, always_2d=False)
    return data, sr

def secure_delete(path: str):
    try:
        if os.path.exists(path):
            length = os.path.getsize(path)
            with open(path, "wb") as f:
                f.write(b"\x00" * length)
            os.remove(path)
    except Exception:
        pass
