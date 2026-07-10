# Single image reused by both the api and frontend services (compose overrides the command).
FROM python:3.11-slim

# System deps: build tools for any wheels that need compiling; curl for healthchecks.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first for better layer caching.
COPY pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir .

# App code.
COPY src ./src
COPY frontend ./frontend

EXPOSE 8000 8501

# Default command; docker-compose overrides per service.
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
