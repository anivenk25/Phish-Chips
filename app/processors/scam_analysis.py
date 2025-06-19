from typing import Dict, Any, List
import subprocess
import json
import os
import openai


def run(cmd: list) -> subprocess.CompletedProcess:
    """
    Wrapper around subprocess.run to execute external commands.
    """
    return subprocess.run(cmd, capture_output=True, text=True)

def process(
    transcription: Dict[str, Any],
    ai_voice: Dict[str, Any],
    emotion: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Analyzes call transcript for scam indicators.
    If the environment variable USE_OPENAI is set, calls OpenAI API;
    otherwise invokes external 'scam_analysis' command.
    Returns a dict with keys: is_scam, confidence, red_flags, target, analysis.
    """
    # OpenAI-based analysis when requested
    if os.getenv("USE_OPENAI"):
        transcript = transcription.get("transcript", "")
        top_emotion = emotion[0].get("top_emotion") if emotion else None
        is_ai = ai_voice.get("is_ai_voice")
        ai_score = ai_voice.get("ai_score")

        system_prompt = '''
You are a scam call analyst. Given a transcript, the detected top emotion, and whether an AI voice was detected,
analyze the content for potential scam indicators. Focus on scam types like phishing, tech support fraud,
IRS scams, fake prizes, and impersonation. Highlight red flags such as urgency, financial or personal data requests.

Return a valid JSON response in the following format:
{
    "is_scam": boolean,
    "confidence": 0-100,
    "red_flags": ["..."],
    "target": ["money"|"personal_info"|"credentials"|"none"],
    "analysis": "..."
}
Only return the JSON. Do not include any additional text.
'''

        user_prompt = f'''
Analyze this call for scam indicators.

Emotion: {top_emotion}
AI Voice Detected: {is_ai} (score: {ai_score})
Transcript: "{transcript[:2000]}"
'''

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                max_tokens=500
            )
            raw = response['choices'][0]['message']['content']
            json_str = raw[raw.find('{'): raw.rfind('}')+1]
            return json.loads(json_str)
        except (openai.error.OpenAIError, json.JSONDecodeError, ValueError) as e:
            return {
                "is_scam": False,
                "confidence": 0,
                "red_flags": [],
                "target": [],
                "analysis": f"Error: {str(e)}"
            }

    # External analysis path
    try:
        result = run(["scam_analysis"])
        raw = result.stdout
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise ValueError("No JSON found")
        json_str = raw[start:end+1]
        return json.loads(json_str)
    except Exception:
        return {
            "is_scam": False,
            "confidence": 0,
            "red_flags": [],
            "target": [],
            "analysis": ""
        }

