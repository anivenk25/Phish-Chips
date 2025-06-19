from typing import Dict, Any, List
import subprocess
import json

def process(
    transcription: Dict[str, Any],
    ai_voice: Dict[str, Any],
    emotion: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Analyzes transcript for scam indicators using a local LLM via Ollama.
    Incorporates AI voice detection and emotion context into the prompt.
    """
    transcript = transcription.get("transcript", "")
    # Extract top emotion if available
    top_emotion = None
    if emotion:
        first = emotion[0]
        top_emotion = first.get("top_emotion")
    # Extract AI voice detection results
    is_ai = ai_voice.get("is_ai_voice")
    ai_score = ai_voice.get("ai_score")
    # Build prompt with context
    prompt = f'''
Analyze this call for scam indicators. Include:
- Emotion: {top_emotion}
- AI Voice Detected: {is_ai} (score: {ai_score})
- Focus on scam types: phishing, tech support, IRS, fake prizes, impersonation
- Red flags: urgency, requests for money/personal info

Respond in this JSON format:
{{
    "is_scam": boolean,
    "confidence": 0-100,
    "red_flags": ["..."],
    "target": ["money"|"personal_info"|"credentials"|"none"],
    "analysis": "..."
}}

Transcript: "{transcript[:2000]}"
'''  
    # Call local LLM via Ollama
    cmd = ["ollama", "run", "hermes3:8b", prompt]
    result = subprocess.run(cmd, capture_output=True, text=True)
    # Parse JSON from LLM output
    try:
        raw = result.stdout
        json_str = raw[raw.find('{'): raw.rfind('}')+1]
        analysis = json.loads(json_str)
        return analysis
    except (json.JSONDecodeError, ValueError, AttributeError):
        # Return empty fallback on parse error
        return {
            "is_scam": False,
            "confidence": 0,
            "red_flags": [],
            "target": [],
            "analysis": ""
        }
