FROM nvidia/cuda:12.4.2-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev \
    ffmpeg libsndfile1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY backend/requirements.txt /app/requirements.txt
RUN pip3 install --upgrade pip \
 && pip3 install -r requirements.txt

# Copy application code
COPY backend /app

# Create directories for recordings and results
RUN mkdir -p /app/recordings /app/results

# Environment variables for directories
ENV RECORDINGS_DIR=/app/recordings
ENV RESULTS_DIR=/app/results

# Default command to start the FastAPI server
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]