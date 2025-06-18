from typing import Dict, Any
import subprocess
import json

def process(transcription: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyzes transcript for scam indicators using a local LLM via Ollama.
    """
    transcript = transcription.get("transcript", "")
    prompt = f'''Analyze this call transcript for scam indicators. Follow these rules STRICTLY:
1. Respond ONLY in valid JSON format
2. Focus on these scam types: phishing, tech support, IRS, fake prizes, impersonation
3. Detect urgency/fear tactics, requests for money/personal info
4. Score confidence based on verbal patterns, not just keywords

{{
    "is_scam": boolean,
    "confidence": 0-100,
    "red_flags": ["tactic_1", "tactic_2"],
    "target": ["money"|"personal_info"|"credentials"|"none"],
    "analysis": "Maximum 2-sentence explanation"
}}

Transcript: "{transcript[:2000]}"'''
    cmd = ["ollama", "run", "hermes3:8b", prompt]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        json_str = result.stdout[result.stdout.find("{"):result.stdout.rfind("}")+1]
        analysis = json.loads(json_str)
        return analysis
    except (json.JSONDecodeError, ValueError):
        return {
            "is_scam": False,
            "confidence": 0,
            "red_flags": [],
            "target": [],
            "analysis": ""
        }
