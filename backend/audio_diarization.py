"""
Speaker diarization using pyannote/speaker-diarization-3.1.
Segments audio by speaker identity and splits into per-speaker WAV files.
"""

import os
import warnings

import torch
from pyannote.audio import Pipeline
from pydub import AudioSegment

warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Deferred model loading for GPU memory carousel
_pipeline = None
_device = None


def _load_pipeline(force_gpu: bool = False):
    """Load the diarization pipeline on demand."""
    global _pipeline, _device

    if _pipeline is not None:
        return

    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN environment variable is required for speaker diarization. "
            "Get a token at https://huggingface.co/settings/tokens and accept "
            "the pyannote/speaker-diarization-3.1 model terms."
        )

    if force_gpu and not torch.cuda.is_available():
        raise RuntimeError("GPU required for diarization but CUDA is not available.")

    if torch.cuda.is_available():
        _device = torch.device("cuda:0")
        torch.cuda.set_device(0)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        print(f"Speaker diarization using GPU: {torch.cuda.get_device_name(0)}")
    else:
        _device = torch.device("cpu")
        print("Speaker diarization falling back to CPU")

    _pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
    ).to(_device)

    print("Diarization pipeline loaded")


def unload_model():
    """Explicitly unload pipeline from GPU memory."""
    global _pipeline, _device
    if _pipeline is not None:
        del _pipeline
        _pipeline = None
    _device = None


def diarize_audio(audio_path: str, use_gpu: bool = True):
    """
    Perform speaker diarization on an audio file.

    Returns:
        List of dicts with keys: speaker, start, end, duration.
    """
    _load_pipeline(force_gpu=use_gpu)

    if torch.cuda.is_available():
        with torch.cuda.device(0):
            diarization = _pipeline(audio_path)
    else:
        diarization = _pipeline(audio_path)

    return [
        {
            "speaker": speaker,
            "start": round(turn.start, 2),
            "end": round(turn.end, 2),
            "duration": round(turn.duration, 2),
        }
        for turn, _, speaker in diarization.itertracks(yield_label=True)
    ]


def split_audio_by_segments(audio_path, segments, output_dir):
    """Split an audio file into per-speaker WAV segments."""
    audio = AudioSegment.from_file(audio_path)
    os.makedirs(output_dir, exist_ok=True)

    for idx, seg in enumerate(segments):
        start_ms = int(seg["start"] * 1000)
        end_ms = int(seg["end"] * 1000)
        speaker = seg["speaker"]
        segment_audio = audio[start_ms:end_ms]
        filename = f"{output_dir}/speaker_{speaker}_{idx + 1}.wav"
        segment_audio.export(filename, format="wav")


if __name__ == "__main__":
    import sys

    audio_file = sys.argv[1] if len(sys.argv) > 1 else "test.wav"
    segments = diarize_audio(audio_file)
    for seg in segments:
        print(
            f"  {seg['speaker']}: {seg['start']:.1f}s - {seg['end']:.1f}s ({seg['duration']:.1f}s)"
        )
