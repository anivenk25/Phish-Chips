import subprocess
import json
import whisper

def transcribe_and_analyze(audio_path):
    print("Loading Whisper model")
    model = whisper.load_model("medium")  

    print("Transcribing and translating audio to English")
    result = model.transcribe(audio_path, task="translate")  
    transcript = result["text"].strip()
    detected_language = result.get("language", "unknown")
    
    print(f"Detected language: {detected_language}")  
    print("Transcription (in English) completed\n")

    prompt = f"""
Analyze this call transcript for scam indicators. Follow these rules STRICTLY:
1. Respond ONLY in valid JSON format
2. Focus on these scam types: phishing, tech support, IRS, fake prizes, impersonation
3. Detect urgency/fear tactics, requests for money/personal info
4. Score confidence based on verbal patterns, not just keywords

{{
    "is_scam": boolean,
    "confidence": 0-100,
    "red_flags": [
        "tactic_1",
        "tactic_2"
    ],
    "target": ["money"|"personal_info"|"credentials"|"none"],
    "analysis": "Maximum 2-sentence explanation"
}}

Transcript: "{transcript[:2000]}"
"""

    print("Running scam analysis using Hermes")
    cmd = ["ollama", "run", "hermes3:8b", prompt]
    result = subprocess.run(cmd, capture_output=True, text=True)

    try:
        # Extract JSON from raw output
        json_str = result.stdout[result.stdout.find('{'):result.stdout.rfind('}')+1]
        analysis = json.loads(json_str)

        print("\n=== 🛡️ SCAM ANALYSIS RESULTS ===\n")
        print(f"📜 Transcript snippet:\n{transcript[:500]}...\n")
        print(f"🔴 Scam detected: {'YES' if analysis.get('is_scam') else 'NO'}")
        print(f"📊 Confidence: {analysis.get('confidence', 0)}%")
        print(f"🎯 Target: {analysis.get('target', 'unknown')}")

        if analysis.get('red_flags'):
            print("\n🚩 Red flags detected:")
            for i, flag in enumerate(analysis['red_flags'][:5], 1):
                print(f" {i}. {flag}")

        print(f"\n💡 Analysis:\n{analysis.get('analysis', 'N/A')}")

    except json.JSONDecodeError:
        print("❌ Failed to parse analysis results")
        print(f"Raw response:\n{result.stdout}")

if __name__ == "__main__":
    audio_file = "/home/r12/Downloads/lol.ogg"  
    transcribe_and_analyze(audio_file)