# Audio Analysis Service - MVP for CyberSec Hackathon

This is a minimal viable product (MVP) for a cyber security hackathon. It provides end-to-end audio analysis to detect potential scams, analyze ambient noise, diarize speakers, detect AI-generated voices, transcribe speech, and identify emotions in audio files.

Features:
- **Scam Analysis**: Uses a local LLM to flag suspicious or scam-like content.
- **Speaker Diarization**: Determines "who spoke when".
- **AI Voice Detection**: Detects if the voice is AI-generated.
- **Transcription**: Transcribes audio using OpenAI Whisper.
- **Emotion Detection**: Identifies emotions in speech.
- **Ambience Classification**: Classifies ambient sounds (e.g., office noises).
- **Simple Web UI**: Upload audio files and view JSON results via browser.

## Quick Start
1. Clone the repository:
   ```bash
   git clone <repo_url>
   cd <repo_directory>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the service:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```
4. Open your browser at http://localhost:8000 to access the web UI.

## Docker
1. Build the image:
   ```bash
   docker build -t audio-analysis-mvp .
   ```
2. Run the container:
   ```bash
   docker run -p 8000:8000 audio-analysis-mvp
   ```
3. Access the web UI at http://localhost:8000.

## API Documentation
- **POST /analyze**: Accepts multipart/form-data with `file` parameter. Supported formats: `.wav`, `.mp3`, `.flac`, `.m4a`, `.ogg`.
- **Response**: JSON containing `ambience`, `diarization`, `ai_voice`, `emotion`, `transcription`, `scam_analysis`, and chunk metadata.

Interactive API docs available at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Security & Privacy
- All uploaded files are securely deleted after processing.
- No audio or transcripts are stored beyond the response.

## License
MIT (for hackathon submission)
