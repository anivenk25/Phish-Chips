from pyannote.audio import Pipeline
import torch
from pydub import AudioSegment
import os
import warnings

# ─── Suppress Warnings ─────────────────────────────
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TF logs if imported

# ─── FORCE GPU-ONLY MODE FOR AZURE ─────────────────
# Check PyTorch CUDA specifically since that's what pyannote uses
if not torch.cuda.is_available():
    print("⚠️ Warning: CUDA not available for PyTorch, diarization may fallback to CPU")
    device = torch.device("cpu")
else:
    # Force GPU device
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    print(f"🎤 Speaker diarization using GPU: {torch.cuda.get_device_name(0)}")

# ─── HF Token from environment variable ─────────────
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# ─── Enable CUDA Optimizations ─────────────────────
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False  # For speed

# ─── Load Pipeline - FORCE GPU ─────────────────────
diarization_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1", use_auth_token=HF_TOKEN
).to(device)

print(f"🎤 Diarization pipeline loaded on GPU")


# ─── Diarization Function ───────────────────────────
def diarize_audio(audio_path: str, use_gpu: bool = True):
    """Perform speaker diarization using GPU when available"""
    # Force GPU processing when available
    if torch.cuda.is_available():
        with torch.cuda.device(0):
            diarization = diarization_pipeline(audio_path)
    else:
        diarization = diarization_pipeline(audio_path)

    return [
        {
            "speaker": speaker,
            "start": round(turn.start, 2),
            "end": round(turn.end, 2),
            "duration": round(turn.duration, 2),
        }
        for turn, _, speaker in diarization.itertracks(yield_label=True)
    ]


# ─── Split Audio by Segments ────────────────────────
def split_audio_by_segments(audio_path, segments, output_dir):
    audio = AudioSegment.from_file(audio_path)
    os.makedirs(output_dir, exist_ok=True)

    for idx, seg in enumerate(segments):
        start_ms = int(seg["start"] * 1000)
        end_ms = int(seg["end"] * 1000)
        speaker = seg["speaker"]
        segment_audio = audio[start_ms:end_ms]
        filename = f"{output_dir}/speaker_{speaker}_{idx + 1}.wav"
        segment_audio.export(filename, format="wav")


# ─── Main ────────────────────────────────────────────
if __name__ == "__main__":
    AUDIO_FILE = "/home/r12/Downloads/lol.ogg"
    OUTPUT_DIR = "diarized_segments"

    print("🔊 Processing...")

    segments = diarize_audio(AUDIO_FILE)
    split_audio_by_segments(AUDIO_FILE, segments, OUTPUT_DIR)

    print("✅ Saved.")
