import os

class Settings:
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "uploads")
    CHUNK_DURATION_SECONDS: float = float(os.getenv("CHUNK_DURATION_SECONDS", 30.0))
    CHUNK_OVERLAP_SECONDS: float = float(os.getenv("CHUNK_OVERLAP_SECONDS", 5.0))
    MAX_WORKERS: int = int(os.getenv("MAX_WORKERS", os.cpu_count()))

settings = Settings()
