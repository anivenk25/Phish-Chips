#!/usr/bin/env python3
"""
Simple VoIP Recording Scanner and Processor
Scans directory for new audio files and processes them in chronological order
"""

import os

# CRITICAL: Force CPU-only for TensorFlow BEFORE any imports
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force TensorFlow to use CPU only
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Reduce TensorFlow warnings
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "false"

import time
import json
import asyncio
import tempfile
import subprocess
from pathlib import Path
from datetime import datetime
import whisper

from ai_voice_detector import detect_deepfake
from background_noise import OfficeAmbienceDetector, convert_to_wav_if_needed
from audio_diarization import diarize_audio, split_audio_by_segments
from emotion_detection import detect_emotion


class VoIPScamScanner:
    def __init__(self):
        # Enable GPU for processing
        self.setup_gpu_environment()

        # Configuration (override via environment variables)
        self.recordings_dir = os.getenv("RECORDINGS_DIR", "/app/recordings")
        self.processed_dir = os.getenv("PROCESSED_DIR", "/app/processed_recordings")
        self.results_dir = os.getenv("RESULTS_DIR", "/app/results")
        self.processed_files_log = os.path.join(self.results_dir, "processed_files.txt")

        # Create directories
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

        # Initialize models as None - load them only when needed to save GPU memory
        print("🔄 Initializing scanner (models loaded on-demand)...")
        self.detector = None
        self.whisper_model = None
        print("✅ Scanner initialized!")

        # Load processed files list
        self.processed_files = self.load_processed_files()

    def setup_gpu_environment(self):
        """Configure environment - GPU only for Ollama LLM, CPU for everything else"""
        print("🔧 TensorFlow forced to CPU-only")

        # Ollama GPU settings - these will be set only when calling Ollama
        self.ollama_gpu_env = {
            "CUDA_VISIBLE_DEVICES": "0",
            "OLLAMA_GPU_LAYERS": "999",
            "OLLAMA_NUM_GPU": "1",
        }

        # Check GPU availability for info (temporarily enable for detection)
        try:
            import torch

            # Temporarily enable GPU for detection
            old_cuda = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"

            if torch.cuda.is_available():
                print(f"🚀 GPU detected: {torch.cuda.get_device_name(0)}")
                print(
                    f"📊 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
                )
                print("💻 GPU reserved exclusively for Ollama LLM")
                print("🔧 All AI models (Whisper, TensorFlow, PyTorch) use CPU")
            else:
                print("⚠️ GPU not available, using CPU for all models")

            # Restore CPU-only setting for PyTorch too
            os.environ["CUDA_VISIBLE_DEVICES"] = old_cuda
        except ImportError:
            print("⚠️ PyTorch not available for GPU detection")

        print("✅ Environment configured: CPU for all AI models, GPU for Ollama only")

    def load_whisper_model(self):
        """Load Whisper model on CPU"""
        if self.whisper_model is None:
            print("🔄 Loading Whisper model (CPU)...")
            self.whisper_model = whisper.load_model("medium", device="cpu")
            print("✅ Whisper model loaded on CPU")
        return self.whisper_model

    def load_noise_detector(self):
        """Load noise detector"""
        if self.detector is None:
            print("🔄 Loading noise detector...")
            self.detector = OfficeAmbienceDetector()
            print("✅ Noise detector loaded")
        return self.detector

    def load_processed_files(self):
        """Load list of already processed files"""
        if os.path.exists(self.processed_files_log):
            with open(self.processed_files_log, "r") as f:
                return set(line.strip() for line in f.readlines())
        return set()

    def save_processed_file(self, filename):
        """Save filename to processed files log"""
        self.processed_files.add(filename)
        with open(self.processed_files_log, "a") as f:
            f.write(f"{filename}\n")

    def get_new_files(self):
        """Get list of new audio files in chronological order"""
        if not os.path.exists(self.recordings_dir):
            print(f"⚠️ Recordings directory not found: {self.recordings_dir}")
            return []

        # Get all audio files
        audio_extensions = (".wav", ".mp3", ".m4a", ".ogg", ".flac")
        all_files = []

        for filename in os.listdir(self.recordings_dir):
            if filename.lower().endswith(audio_extensions):
                file_path = os.path.join(self.recordings_dir, filename)
                if filename not in self.processed_files and os.path.isfile(file_path):
                    # Get file creation time
                    created_time = os.path.getctime(file_path)
                    all_files.append((created_time, filename, file_path))

        # Sort by creation time (chronological order)
        all_files.sort(key=lambda x: x[0])

        return [(filename, file_path) for _, filename, file_path in all_files]

    async def analyze_audio_file(self, file_path):
        """Analyze a single audio file"""
        filename = Path(file_path).name
        start_time = time.time()

        print(f"\n🔍 Analyzing: {filename}")
        print(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        try:
            # Convert to WAV if needed
            wav_path = convert_to_wav_if_needed(file_path)

            # Quick audio validation
            import librosa

            try:
                # Test if audio can be loaded
                y_test, sr_test = librosa.load(wav_path, sr=None, duration=0.1)
                if len(y_test) == 0:
                    print(f"⚠️ Audio file appears to be empty or corrupted")
                    return None
            except Exception as audio_error:
                print(f"⚠️ Cannot load audio file: {audio_error}")
                return None

            # Load models - all on CPU (skip noise detector to avoid TensorFlow GPU conflicts)
            print("🔄 Loading models for analysis...")
            whisper_model = self.load_whisper_model()

            # Run analyses on CPU
            print("🔍 Running deepfake detection (CPU)...")
            is_cloned, clone_conf = detect_deepfake(wav_path)
            print(
                f"🤖 Deepfake result: cloned={is_cloned}, confidence={clone_conf:.2f}"
            )

            print("🔍 Running background noise analysis (CPU)...")
            # TEMPORARY: Skip background noise to avoid TensorFlow GPU conflicts
            print(
                "⚠️ Background noise analysis temporarily disabled to prevent GPU conflicts"
            )
            noise_res = {
                "is_office": True,
                "confidence": 0.5,
                "detected_tags": [],
                "composite_score": 0.5,
                "strong_signal": False,
            }
            print(f"🔊 Noise analysis complete")

            print("🔍 Running speaker diarization (CPU)...")
            diarization_res = diarize_audio(wav_path)
            print(f"👥 Found {len(diarization_res)} speaker segments")

            print("🔍 Running speech transcription (CPU)...")
            whisper_res = whisper_model.transcribe(wav_path, task="translate")
            transcript = whisper_res.get("text", "").strip()
            print(
                f"📝 Transcript: {transcript[:100]}{'...' if len(transcript) > 100 else ''}"
            )

            # Emotion detection per speaker segment (CPU)
            emotions = []
            if diarization_res:
                print("🔍 Running emotion detection...")
                with tempfile.TemporaryDirectory() as seg_dir:
                    split_audio_by_segments(wav_path, diarization_res, seg_dir)
                    for idx, seg in enumerate(diarization_res):
                        seg_file = os.path.join(
                            seg_dir, f"speaker_{seg['speaker']}_{idx + 1}.wav"
                        )
                        if os.path.exists(seg_file):
                            emotion_result = detect_emotion(seg_file)
                            emotions.append(emotion_result)

            # Normalize noise data

            # Normalize noise data
            try:
                noise_data = {
                    "is_office": bool(noise_res.get("is_office")),
                    "confidence": float(noise_res.get("confidence", 0)),
                    "detected_tags": noise_res.get("detected_tags", []),
                    "composite_score": float(noise_res.get("composite_score", 0)),
                    "strong_signal": bool(noise_res.get("strong_signal", False)),
                }
            except Exception:
                noise_data = noise_res if noise_res else {}

            # Run LLM analysis for scam detection
            scam_analysis = await self.analyze_for_scams(
                transcript, is_cloned, clone_conf, noise_data, diarization_res, emotions
            )

            # Compile results
            result = {
                "filename": filename,
                "file_path": file_path,
                "timestamp": datetime.now().isoformat(),
                "processing_time": time.time() - start_time,
                "transcript": transcript,
                "is_cloned": is_cloned,
                "clone_confidence": clone_conf,
                "noise_analysis": noise_data,
                "diarization": diarization_res,
                "emotions": emotions,
                "scam_analysis": scam_analysis,
            }

            # Save results
            result_file = os.path.join(
                self.results_dir, f"analysis_{Path(filename).stem}.json"
            )
            with open(result_file, "w") as f:
                json.dump(result, f, indent=2)

            # Move processed file
            processed_file = os.path.join(self.processed_dir, filename)
            try:
                os.rename(file_path, processed_file)
            except OSError:
                # If move fails, copy and remove original
                import shutil

                shutil.copy2(file_path, processed_file)
                os.remove(file_path)

            # Mark as processed
            self.save_processed_file(filename)

            # Print summary
            print(f"✅ Analysis complete!")
            if scam_analysis.get("is_scam"):
                print(
                    f"🚨 SCAM DETECTED! Confidence: {scam_analysis.get('confidence', 0)}%"
                )
                print(
                    f"🔴 Risk factors: {', '.join(scam_analysis.get('red_flags', []))}"
                )
            else:
                print(
                    f"✅ Clean call (confidence: {scam_analysis.get('confidence', 0)}%)"
                )
            print(f"⏱️ Processing time: {result['processing_time']:.1f}s")

            return result

        except Exception as e:
            print(f"❌ Error analyzing {filename}: {e}")
            return None

    async def analyze_for_scams(
        self, transcript, is_cloned, clone_conf, noise_data, diarization_res, emotions
    ):
        """Use LLM to analyze for scam indicators"""

        # Simple rule-based fallback analysis
        scam_score = 0
        red_flags = []

        # Voice cloning indicator
        if is_cloned and clone_conf > 0.7:
            scam_score += 40
            red_flags.append("Potential voice cloning detected")

        # Background noise indicators
        if not noise_data.get("is_office", True):
            scam_score += 15
            red_flags.append("Unusual background environment")

        # Multiple speakers (confusion tactic)
        if len(diarization_res) > 2:
            scam_score += 20
            red_flags.append("Multiple speakers detected")

        # Transcript analysis for scam keywords
        scam_keywords = [
            "urgent",
            "immediately",
            "suspended",
            "verify",
            "social security",
            "bank account",
            "credit card",
            "confirm",
            "expires",
            "locked",
            "police",
            "arrest",
            "warrant",
            "IRS",
            "tax",
            "refund",
            "prize",
            "winner",
            "congratulations",
            "click",
            "link",
            "password",
        ]

        transcript_lower = transcript.lower()
        keyword_matches = [kw for kw in scam_keywords if kw in transcript_lower]
        if keyword_matches:
            scam_score += len(keyword_matches) * 5
            red_flags.append(
                f"Scam keywords detected: {', '.join(keyword_matches[:3])}"
            )

        # Emotion analysis (high stress/fear)
        if emotions:
            for emotion in emotions:
                if isinstance(emotion, dict):
                    top_emotion = emotion.get("top_emotion", "")
                    if top_emotion in ["angry", "fearful", "stressed"]:
                        scam_score += 10
                        red_flags.append(f"High-stress emotion detected: {top_emotion}")

        # Try LLM analysis if available
        try:
            prompt = f"""
Analyze this call for scam indicators. Respond ONLY with valid JSON:
{{
    "is_scam": boolean,
    "confidence": 0-100,
    "red_flags": ["flag1", "flag2"],
    "reasoning": "brief explanation"
}}

Call data:
- Transcript: "{transcript[:1000]}"
- Voice cloned: {is_cloned} (confidence: {clone_conf:.2f})
- Speakers: {len(diarization_res)}
- Background: {"office" if noise_data.get("is_office") else "non-office"}
"""

            # Run Ollama with GPU acceleration
            print("🔍 Running LLM analysis (GPU)...")
            env = os.environ.copy()

            # Explicitly set GPU environment for Ollama
            env["CUDA_VISIBLE_DEVICES"] = "0"  # Enable GPU for this subprocess
            env["OLLAMA_GPU_LAYERS"] = "999"  # Use all GPU layers
            env["OLLAMA_NUM_GPU"] = "1"  # Use 1 GPU

            print(
                f"🔧 Ollama GPU env: CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES')}"
            )

            start_llm = time.time()
            proc = subprocess.run(
                ["ollama", "run", "hermes3:8b", prompt],
                capture_output=True,
                text=True,
                timeout=30,
                env=env,  # Pass environment with GPU settings
            )
            llm_time = time.time() - start_llm

            if proc.returncode == 0:
                json_str = proc.stdout[
                    proc.stdout.find("{") : proc.stdout.rfind("}") + 1
                ]
                llm_result = json.loads(json_str)
                print(
                    f"🤖 LLM Analysis: {'SCAM' if llm_result.get('is_scam') else 'Clean'} ({llm_result.get('confidence', 0)}%) - {llm_time:.1f}s"
                )
                return llm_result
            else:
                print(f"⚠️ LLM process failed: {proc.stderr}")

        except Exception as e:
            print(f"⚠️ LLM analysis failed, using rule-based analysis: {e}")

        # Fallback to rule-based analysis
        return {
            "is_scam": scam_score > 50,
            "confidence": min(scam_score, 100),
            "red_flags": red_flags,
            "reasoning": f"Rule-based analysis with {len(red_flags)} risk factors detected",
        }

    async def scan_and_process(self):
        """Main scanning loop"""
        print("🎯 VoIP Scam Scanner started!")
        print(f"📁 Monitoring: {self.recordings_dir}")
        print(f"📊 Results: {self.results_dir}")

        while True:
            try:
                # Get new files to process
                new_files = self.get_new_files()

                if new_files:
                    print(f"\n📋 Found {len(new_files)} new files to process")

                    for filename, file_path in new_files:
                        await self.analyze_audio_file(file_path)
                        # Small delay between files
                        await asyncio.sleep(1)
                else:
                    # No new files, wait before checking again
                    print(".", end="", flush=True)  # Progress indicator
                    await asyncio.sleep(5)

            except KeyboardInterrupt:
                print("\n🛑 Scanner stopped by user")
                break
            except Exception as e:
                print(f"\n❌ Scanner error: {e}")
                await asyncio.sleep(10)


async def main():
    scanner = VoIPScamScanner()
    await scanner.scan_and_process()


if __name__ == "__main__":
    print("🚀 Starting VoIP Scam Detection Scanner...")
    asyncio.run(main())
