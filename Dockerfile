FROM python:3.11-slim

# ffmpeg is required for audio extraction and clip cutting
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY main.py .

# Render/Railway set $PORT automatically; default to 8000 for local docker run
ENV PORT=8000
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT}
