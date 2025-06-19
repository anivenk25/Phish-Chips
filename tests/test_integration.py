import os
import sys
import pytest

# Ensure project root is in PATH so 'app' module can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.main import AnalysisService, ambience, diarization, ai_voice, emotion, transcription, scam_analysis


@pytest.fixture
def dummy_wav(tmp_path):
    # Create a dummy WAV file (empty content, stub processors will not read it)
    wav_file = tmp_path / "test.wav"
    wav_file.write_bytes(b"")
    return str(wav_file)


def test_analysis_service_integration(monkeypatch, dummy_wav):
    # Prepare stub outputs for each processor
    expected_diarization = [
        {"speaker": "Speaker1", "start": 0.0, "end": 1.0, "duration": 1.0}
    ]
    expected_ai_voice = {"is_ai_voice": False, "ai_score": 0.1234}
    expected_ambience = {
        "is_office": True,
        "confidence": 0.75,
        "detected_tags": ["Typing"],
        "composite_score": 0.75
    }
    expected_emotion = [
        {"top_emotion": "happy", "emotion_probs": {"happy": 0.8, "sad": 0.2}}
    ]
    expected_transcription = {"transcript": "hello world", "segments": []}
    expected_scam = {"is_scam": False, "confidence": 0, "red_flags": [], "target": [], "analysis": ""}

    # Monkey-patch processor functions to return stub outputs
    monkeypatch.setattr(diarization, 'process', lambda file_path, metadata: expected_diarization)
    monkeypatch.setattr(ai_voice,   'process', lambda file_path, metadata: expected_ai_voice)
    monkeypatch.setattr(ambience,    'process', lambda file_path: expected_ambience)
    monkeypatch.setattr(emotion,     'process', lambda file_path, metadata: expected_emotion)
    monkeypatch.setattr(transcription, 'process', lambda file_path, metadata: expected_transcription)
    monkeypatch.setattr(scam_analysis, 'process', lambda trans, ai, emo: expected_scam)

    # Run the AnalysisService with dummy input and empty metadata
    service = AnalysisService(dummy_wav, metadata=[])
    results = service.run()

    # Verify that the results aggregate all processor outputs correctly
    assert results["diarization"] == expected_diarization
    assert results["ai_voice"]   == expected_ai_voice
    assert results["ambience"]    == expected_ambience
    assert results["emotion"]     == expected_emotion
    assert results["transcription"] == expected_transcription
    assert results["scam_analysis"] == expected_scam
    
def test_analyze_audio_endpoint(monkeypatch, tmp_path):
    """
    Integration test for the /analyze FastAPI endpoint, with stubbed AnalysisService.run.
    """
    # Create a short silent WAV file for upload (1 second of silence at 16 kHz)
    import numpy as np
    import soundfile as sf
    from fastapi.testclient import TestClient
    from app.main import app, AnalysisService

    sr = 16000
    duration = 1.0
    samples = int(sr * duration)
    data = np.zeros(samples, dtype='float32')
    wav_path = tmp_path / "silent.wav"
    sf.write(str(wav_path), data, sr)

    # Stub the AnalysisService.run to return fixed results
    stub_results = {
        "ambience": {"stub": True},
        "diarization": [],
        "ai_voice": {"stub": True},
        "emotion": [],
        "transcription": {"stub": True},
        "scam_analysis": {"stub": True}
    }
    monkeypatch.setattr(AnalysisService, 'run', lambda self: stub_results)

    client = TestClient(app)
    with open(wav_path, 'rb') as f:
        response = client.post(
            "/analyze",
            files={"file": ("silent.wav", f, "audio/wav")}
        )
    assert response.status_code == 200, response.text
    json_data = response.json()
    # The response should include stub_results and chunks metadata
    for key, value in stub_results.items():
        assert key in json_data, f"Missing key {key} in response"
        assert json_data[key] == value
    # Check that chunks metadata is returned and has at least one chunk
    assert 'chunks' in json_data
    chunks = json_data['chunks']
    assert isinstance(chunks, list) and len(chunks) >= 1
    # Check chunk fields
    first = chunks[0]
    assert 'index' in first and 'start_time' in first and 'end_time' in first