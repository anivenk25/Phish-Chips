# Use slim Python base
FROM python:3.10-slim

# Install uv and required system deps
RUN apt-get update && apt-get install -y curl build-essential \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy code
COPY . .

# Install project dependencies using uv
RUN uv venv && \
    .venv/bin/uv pip install -e . && \
    .venv/bin/uv pip install "uvicorn[standard]" fastapi

# Expose port for FastAPI (default 8000)
EXPOSE 8000

# Start FastAPI app from app/main.py
CMD [".venv/bin/uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

