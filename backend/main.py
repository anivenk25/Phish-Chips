#!/usr/bin/env python3

# Set CUDA memory optimization BEFORE importing other modules
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Better error messages

import json
import time
import asyncio
import tempfile
import subprocess
import psutil
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Dict, Any, List

import torch
import whisper

from ai_voice_detector import detect_deepfake, unload_model as unload_deepfake
from background_noise import OfficeAmbienceDetector, convert_to_wav_if_needed
from audio_diarization import (
    diarize_audio,
    split_audio_by_segments,
    unload_model as unload_diarization,
)
from emotion_detection import detect_emotion, unload_model as unload_emotion

# FORCE GPU-ONLY MODE FOR AZURE GPU INSTANCE
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use only GPU 0
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed


def enforce_gpu_only():
    """Enforce GPU-only mode for all operations"""
    if not torch.cuda.is_available():
        raise RuntimeError("❌ CUDA not available! This system requires GPU.")

    # Set device to GPU only
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)

    # Clear any CPU fallbacks
    torch.set_default_tensor_type("torch.cuda.FloatTensor")

    print(f"🚀 GPU-ONLY MODE: {torch.cuda.get_device_name(0)}")
    print(
        f"📊 GPU Memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3}GB"
    )

    return device


def run_ollama_analysis(prompt: str) -> dict:
    """Start Ollama, run analysis, then immediately kill process to free GPU"""
    print("🔄 Starting Ollama for LLM analysis...")

    try:
        # Start Ollama process
        start_llm = time.time()
        proc = subprocess.run(
            ["ollama", "run", "hermes3:8b", prompt],
            capture_output=True,
            text=True,
            timeout=60,  # Longer timeout for model loading
        )
        llm_time = time.time() - start_llm

        # IMMEDIATELY kill all Ollama processes before processing output
        print("🔪 IMMEDIATELY killing Ollama processes to free GPU...")
        kill_count = 0
        for process in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                if "ollama" in process.info["name"].lower():
                    process.kill()
                    kill_count += 1
                    print(f"   ✅ Killed Ollama process PID: {process.info['pid']}")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        # Also kill via pkill as backup
        try:
            subprocess.run(["pkill", "-f", "ollama"], check=False)
        except:
            pass

        print(f"   🧹 Killed {kill_count} Ollama processes, GPU memory freed")

        if proc.returncode == 0:
            # Extract JSON from output
            json_str = proc.stdout[proc.stdout.find("{") : proc.stdout.rfind("}") + 1]
            llm_result = json.loads(json_str)
            print(f"🤖 LLM Analysis complete: {llm_time:.1f}s")

            return llm_result
        else:
            print(f"⚠️ Ollama failed: {proc.stderr}")
            return None

    except Exception as e:
        print(f"⚠️ Ollama error: {e}")

        # Kill Ollama processes even on error
        try:
            for process in psutil.process_iter(["pid", "name"]):
                if "ollama" in process.info["name"].lower():
                    process.kill()
            subprocess.run(["pkill", "-f", "ollama"], check=False)
        except:
            pass

        return None


def convert_raw_voip_to_wav(file_path: str) -> str:
    """Convert raw VoIP recording to proper WAV format"""
    try:
        # Read as raw binary data
        with open(file_path, "rb") as f:
            raw_data = f.read()

        # Convert to numpy array (assuming 8kHz 16-bit signed integer - common for VoIP)
        audio_data = np.frombuffer(raw_data, dtype=np.int16)

        # Normalize to float32
        normalized_data = audio_data.astype(np.float32) / np.iinfo(np.int16).max

        # Create temporary WAV file
        temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_wav.close()

        # Save as proper WAV file (8kHz is typical for VoIP)
        sf.write(temp_wav.name, normalized_data, 8000)

        return temp_wav.name

    except Exception as e:
        print(f"    ⚠️ Raw conversion failed: {e}")
        return file_path  # Return original if conversion fails


def clear_gpu_memory():
    """Aggressively clear GPU memory between batches - GPU ONLY"""
    import gc

    if not torch.cuda.is_available():
        raise RuntimeError("❌ GPU required for all operations!")

    # Force garbage collection first
    gc.collect()

    # Clear PyTorch cache
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # Additional cleanup for better memory management
    torch.cuda.ipc_collect()

    # Force another garbage collection
    gc.collect()

    memory_allocated = torch.cuda.memory_allocated(0) // 1024**2
    memory_reserved = torch.cuda.memory_reserved(0) // 1024**2
    total_memory = torch.cuda.get_device_properties(0).total_memory // 1024**2
    free_memory = total_memory - memory_reserved

    print(f"    🧹 GPU Memory: {memory_allocated}MB allocated, {free_memory}MB free")

    # If we're using too much memory, wait a bit for cleanup
    if memory_allocated > 3000:  # More than 3GB
        print(f"    ⚠️ High GPU usage ({memory_allocated}MB), waiting for cleanup...")
        time.sleep(2)
        torch.cuda.empty_cache()
        gc.collect()


def get_gpu_device():
    """Get GPU device - GPU ONLY MODE"""
    if not torch.cuda.is_available():
        raise RuntimeError("❌ GPU required! CUDA not available.")

    device = torch.device("cuda:0")
    print(f"    � Using GPU: {torch.cuda.get_device_name(0)}")
    return device


async def analyze_audio_file(file_path: str) -> Dict[str, Any]:
    """Analyze a single audio file with optimized GPU batch processing"""
    start_time = time.time()
    print(f"\n🔍 Analyzing: {Path(file_path).name}")

    try:
        # First try to convert from raw VoIP format
        print("  🔄 Converting raw VoIP recording to WAV...")
        wav_path = convert_raw_voip_to_wav(file_path)

        # If that failed, try the normal conversion
        if wav_path == file_path:
            print("  🔄 Trying standard audio conversion...")
            wav_path = convert_to_wav_if_needed(file_path)

        loop = asyncio.get_event_loop()

        # BATCH 1: AI Clone Detection + Whisper + Background Noise (GPU ONLY)
        print("  🎯 BATCH 1: Clone Detection + Whisper + Background Noise (GPU ONLY)")

        # Get GPU device (enforce GPU-only)
        device = get_gpu_device()

        # Load models for batch 1 with MANDATORY GPU usage
        detector = OfficeAmbienceDetector()

        # Load Whisper with FORCED GPU device (using small model for better memory management)
        print("    📥 Loading Whisper model on GPU...")
        whisper_model = whisper.load_model("small", device=device)

        # Verify model is on GPU
        if hasattr(whisper_model, "device"):
            print(f"    ✅ Whisper model confirmed on: {whisper_model.device}")
        else:
            print(f"    ✅ Whisper model loaded on: {device}")

        # Force all operations to use GPU
        clone_task = loop.run_in_executor(
            None, lambda: detect_deepfake(wav_path, force_gpu=True)
        )
        noise_task = loop.run_in_executor(None, detector.detect_office, wav_path)
        diarization_task = loop.run_in_executor(
            None, lambda: diarize_audio(wav_path, use_gpu=True)
        )
        whisper_task = loop.run_in_executor(
            None,
            lambda: whisper_model.transcribe(wav_path, task="translate", fp16=True),
        )

        # Gather batch 1 results
        (
            (is_cloned, clone_conf),
            noise_res,
            diarization_res,
            whisper_res,
        ) = await asyncio.gather(clone_task, noise_task, diarization_task, whisper_task)

        transcript = whisper_res.get("text", "").strip()
        print(
            f"    📝 Transcript: {transcript[:100]}..."
            if len(transcript) > 100
            else f"    📝 Transcript: {transcript}"
        )
        print(f"    🤖 Clone confidence: {clone_conf:.2f}")
        print(
            f"    👥 Speakers detected: {len(set(seg['speaker'] for seg in diarization_res))}"
        )

        # Aggressively clear GPU memory after batch 1
        del detector, whisper_model
        unload_deepfake()
        unload_diarization()
        # Force Python to release references
        import gc

        gc.collect()
        clear_gpu_memory()
        print(f"    🗑️ Batch 1 models unloaded from GPU")

        # BATCH 2: Emotion Detection on Diarized Segments (GPU ONLY)
        print("  🎯 BATCH 2: Emotion Detection (GPU ONLY)")
        emotions = []
        with tempfile.TemporaryDirectory() as seg_dir:
            split_audio_by_segments(wav_path, diarization_res, seg_dir)
            tasks = []
            for idx, seg in enumerate(diarization_res):
                seg_file = os.path.join(
                    seg_dir, f"speaker_{seg['speaker']}_{idx + 1}.wav"
                )
                # Force GPU usage for emotion detection
                tasks.append(
                    loop.run_in_executor(
                        None, lambda f=seg_file: detect_emotion(f, use_gpu=True)
                    )
                )
            emotions = await asyncio.gather(*tasks)

        print(f"    😊 Emotions analyzed for {len(emotions)} segments on GPU")

        # Aggressively clear GPU memory after batch 2
        unload_emotion()
        import gc

        gc.collect()
        clear_gpu_memory()
        print(f"    🗑️ Batch 2 models unloaded from GPU")

        # BATCH 3: LLM Analysis via Subprocess (GPU ONLY)
        print("  🎯 BATCH 3: LLM Scam Analysis (GPU ONLY)")

        # Normalize analysis results for LLM
        try:
            noise_data = {
                "is_office": bool(noise_res.get("is_office")),
                "confidence": float(noise_res.get("confidence")),
                "detected_tags": noise_res.get("detected_tags"),
                "composite_score": float(noise_res.get("composite_score")),
                "strong_signal": bool(noise_res.get("strong_signal")),
            }
        except Exception:
            noise_data = noise_res

        # Build LLM prompt with all analysis results
        prompt = f"""
You are an AI assistant specialized in detecting VoIP scam calls.

Given the following analysis data from a VoIP call recording, determine whether this is a scam call. Respond ONLY with a single JSON object containing these keys:
  scam (boolean)
  reasoning (string)
  confidence_score (float between 0.0 and 1.0)

VoIP Call Analysis Data:
{{
  "is_cloned": {is_cloned},
  "clone_confidence": {clone_conf:.2f},
  "noise": {json.dumps(noise_data)},
  "diarization": {json.dumps(diarization_res)},
  "emotions": {json.dumps(emotions)},
  "transcript": "{transcript}"
}}

Consider these scam indicators:
- Voice cloning/deepfake detection
- Pressure tactics in conversation
- Unusual background noise patterns
- Multiple speakers trying to confuse
- Emotional manipulation
- Requests for personal information
"""

        # Run LLM analysis with subprocess and immediate cleanup
        llm_out = run_ollama_analysis(prompt)

        if llm_out is None:
            print("    ⚠️ LLM analysis failed, using fallback")
            # Fallback analysis based on detection results
            scam_score = 0.0
            if is_cloned and clone_conf > 0.7:
                scam_score += 0.4
            if not noise_data.get("is_office", True):
                scam_score += 0.2
            if len(diarization_res) > 2:  # Multiple speakers
                scam_score += 0.2

            llm_out = {
                "scam": scam_score > 0.5,
                "reasoning": f"Automated analysis: clone_confidence={clone_conf:.2f}, multiple_speakers={len(diarization_res) > 2}",
                "confidence_score": scam_score,
            }

        # Build final result with all data
        processing_time = time.time() - start_time
        result = {
            "file_path": file_path,
            "file_name": Path(file_path).name,
            "processing_time": processing_time,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "scam": llm_out.get("scam", False),
            "reasoning": llm_out.get("reasoning", "Analysis completed"),
            "confidence_score": llm_out.get("confidence_score", 0.5),
            "is_cloned": is_cloned,
            "clone_confidence": clone_conf,
            "noise": noise_data,
            "diarization": diarization_res,
            "emotions": emotions,
            "transcript": transcript,
        }

        print(
            f"  ✅ Analysis complete: {'🚨 SCAM DETECTED' if result['scam'] else '✅ Clean call'} (confidence: {result['confidence_score']:.2f})"
        )
        print(f"  ⏱️ Processing time: {processing_time:.2f}s")

        # Delete source audio after successful analysis (privacy)
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"  🗑️ Source audio deleted: {Path(file_path).name}")
            # Also remove any temporary WAV conversion
            if wav_path != file_path and os.path.exists(wav_path):
                os.remove(wav_path)
        except OSError as del_err:
            print(f"  ⚠️ Could not delete source audio: {del_err}")

        return result

    except Exception as e:
        print(f"  ❌ Error analyzing {file_path}: {e}")
        return {
            "file_path": file_path,
            "file_name": Path(file_path).name,
            "error": str(e),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    finally:
        # Ensure GPU memory is cleared even on error
        clear_gpu_memory()


def unload_whisper_model():
    """Unload Whisper model from GPU memory"""
    global whisper_model
    if "whisper_model" in globals() and whisper_model is not None:
        del whisper_model
        whisper_model = None
        print(f"    🗑️ Whisper model unloaded from GPU")
        clear_gpu_memory()


def unload_all_models():
    """Unload all models from GPU memory"""
    # This will be called by individual detector classes
    clear_gpu_memory()
    print(f"    🧹 All models cleared from GPU memory")


async def main():
    """Main processing function with STRICT GPU-ONLY processing for Azure"""
    print("🚀 VoIP Scam Detection - AZURE GPU-ONLY MODE")
    print("💪 CUDA 12.4 Optimized - ALL components forced to GPU")
    print("🔧 GPU Strategy: Sequential batches with aggressive memory management")
    print("   📋 Batch 1: Clone Detection + Whisper + Background Noise (GPU)")
    print("   📋 Batch 2: Emotion Detection on Diarized Segments (GPU)")
    print("   📋 Batch 3: LLM Analysis via Subprocess (GPU)")

    # ENFORCE GPU-ONLY MODE AT STARTUP
    print("\n🚀 ENFORCING GPU-ONLY MODE...")
    try:
        device = enforce_gpu_only()

        # Kill any existing Ollama processes
        for process in psutil.process_iter(["pid", "name"]):
            try:
                if "ollama" in process.info["name"].lower():
                    process.kill()
                    print(
                        f"   🔪 Killed existing Ollama process PID: {process.info['pid']}"
                    )
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        # Clear GPU memory aggressively
        clear_gpu_memory()

        # Verify GPU status for Azure deployment
        memory_free = torch.cuda.get_device_properties(
            0
        ).total_memory - torch.cuda.memory_reserved(0)
        print(f"   🚀 Azure GPU ready: {memory_free // 1024**3:.1f}GB free memory")
        print(f"   📊 CUDA Version: {torch.version.cuda}")

    except Exception as e:
        print(f"   ❌ GPU setup failed: {e}")
        raise RuntimeError("Azure GPU instance required for this application!")

    # Configuration for deployment (override via environment variables)
    RECORDINGS_DIR = os.getenv("RECORDINGS_DIR", "/app/recordings")
    RESULTS_DIR = os.getenv("RESULTS_DIR", "/app/results")

    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Check if recordings directory exists
    if not os.path.exists(RECORDINGS_DIR):
        print(f"❌ Recordings directory not found: {RECORDINGS_DIR}")
        print("Make sure your VoIP server has created recordings!")
        return

    print("📊 Models will be loaded per batch for optimal GPU usage")

    # Get all audio files sorted chronologically (oldest first)
    audio_extensions = (".wav", ".mp3", ".m4a", ".ogg", ".flac")
    audio_files = []

    for file_name in os.listdir(RECORDINGS_DIR):
        if file_name.lower().endswith(audio_extensions):
            file_path = os.path.join(RECORDINGS_DIR, file_name)
            try:
                # Get file creation time
                creation_time = os.path.getctime(file_path)
                audio_files.append((creation_time, file_path))
            except OSError:
                continue

    # Sort by creation time (chronological order)
    audio_files.sort(key=lambda x: x[0])

    if not audio_files:
        print(f"📁 No audio files found in {RECORDINGS_DIR}")
        return

    print(f"📂 Found {len(audio_files)} audio files to process")
    print("🕒 Processing in chronological order (oldest first)...")

    # Process each file with optimized GPU batching
    all_results = []
    for i, (creation_time, file_path) in enumerate(audio_files, 1):
        creation_date = time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(creation_time)
        )
        print(f"\n[{i}/{len(audio_files)}] File created: {creation_date}")

        result = await analyze_audio_file(file_path)
        all_results.append(result)

        # Save individual result
        result_file = os.path.join(RESULTS_DIR, f"analysis_{Path(file_path).stem}.json")
        with open(result_file, "w") as f:
            json.dump(result, f, indent=2)

    # Save summary report
    summary = {
        "total_files": len(audio_files),
        "processed_files": len([r for r in all_results if "error" not in r]),
        "failed_files": len([r for r in all_results if "error" in r]),
        "scam_detected": len([r for r in all_results if r.get("scam", False)]),
        "processing_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": all_results,
    }

    summary_file = os.path.join(RESULTS_DIR, "processing_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    # Print final summary
    print(f"\n📊 Processing Summary:")
    print(f"  📁 Total files: {summary['total_files']}")
    print(f"  ✅ Processed: {summary['processed_files']}")
    print(f"  ❌ Failed: {summary['failed_files']}")
    print(f"  🚨 Scams detected: {summary['scam_detected']}")
    print(f"  💾 Results saved to: {RESULTS_DIR}")

    if summary["scam_detected"] > 0:
        print(f"\n🚨 SCAM ALERTS:")
        for result in all_results:
            if result.get("scam", False):
                print(
                    f"  🚨 {result['file_name']} - {result.get('reasoning', 'Scam detected')}"
                )


if __name__ == "__main__":
    asyncio.run(main())
