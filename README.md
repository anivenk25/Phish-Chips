 # VoIP Scam Detection Backend
 
 This repository contains the **VoIP Scam Detection** backend service, which processes VoIP call recordings and uses AI models to detect potential scam calls.
 
 ## Features
 - Voice cloning (deepfake) detection
 - Speech-to-text transcription (Whisper)
 - Background noise analysis
 - Speaker diarization
 - Emotion detection
 - LLM-based scam analysis
 
 ## Prerequisites
 - NVIDIA GPU with CUDA 12.4 and Nvidia Container Toolkit installed
 - Docker and Docker Compose
 
 ## Getting Started with Docker
 1. Clone this repository:
    ```bash
    git clone <repo_url>
    cd <repo_directory>
    ```
 
 2. Build the Docker image:
    ```bash
    docker-compose build
    ```
 
 3. Create host directories for audio files and results:
    ```bash
    mkdir recordings results
    ```
 
 4. Run the service:
   ```bash
   docker-compose up --build
   ```

   This will start two services:
   - **Backend API** on port **8000** (FastAPI + GPU analysis)
   - **Signaling Server** on port **3000** (receives uploads and notifies backend)

   - Place audio files (`.wav`, `.mp3`, etc.) into the `recordings` directory
   - Processed results will be saved as JSON files in the `results` directory

## Uploading Recordings

Use `curl` or your VoIP client to upload recordings to the signaling server:

```bash
curl -X POST -F "recording=@/path/to/test_audio.wav" http://localhost:3000/upload-recording
```

This endpoint will:
1. Save the file into `recordings/`
2. Trigger backend analysis via `/force-analyze/{filename}`
3. Store the analysis result in the `results/` directory

## Accessing Results

- **Host directory**: View individual JSON result files in `results/`
- **API endpoint**: Fetch all results via the backend:

  ```bash
  curl http://localhost:8000/analysis-results
  ```
 
 ## Configuration
- `RECORDINGS_DIR`: Path to input audio files inside the container (default: `/recordings`)
- `RESULTS_DIR`: Path to output results inside the container (default: `/results`)
 
 These can be overridden by setting environment variables in `docker-compose.yml` or via the Docker command line.
 
 ## Running Locally (Without Docker)
 1. Install Python dependencies:
    ```bash
    pip install --upgrade pip
    pip install -r backend/requirements.txt
    ```
 
 2. Set environment variables (optional):
    ```bash
    export RECORDINGS_DIR=/path/to/recordings
    export RESULTS_DIR=/path/to/results
    ```
 
 3. Run the analysis service:
    ```bash
    python3 backend/main.py
    ```
 
 ## Notes
 - The service enforces GPU-only mode by default and will fail if no compatible CUDA GPU is found.
 - Models are loaded and unloaded in batches to optimize GPU memory usage.
 
 ---
 *End of README*