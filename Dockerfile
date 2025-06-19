# Use slim Python base
FROM python:3.9-slim

# Install uv and required system deps
RUN apt-get update && apt-get install -y --no-install-recommends curl wget build-essential \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Add uv to PATH (default install location)
ENV PATH="/root/.local/bin:${PATH}"

# Set working directory
WORKDIR /app

# Copy code
COPY . .

# Install project dependencies using uv
RUN uv venv && \
    uv pip install -e . && \
    uv pip install -r requirements.txt &&\
    uv pip install "uvicorn[standard]" fastapi

# Set PATH so CMD finds uvicorn in the venv
ENV PATH="/app/.venv/bin:$PATH"

# Start FastAPI app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

