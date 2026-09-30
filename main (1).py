"""
Clip Finder backend (upload-based)
-----------------------------------
User uploads a video file directly (already downloaded from YouTube/Kick/wherever).
This service:
  1. Saves the uploaded file
  2. Extracts audio and transcribes it with timestamps (Groq's free cloud Whisper API —
     no local model, so this runs fine even on a 512MB free server)
  3. Sends the transcript to Gemini (free tier) to find the best 30-40s highlight moments
  4. Cuts those moments out of the video with ffmpeg
  5. Returns downloadable clip files

No yt-dlp, no link-downloading, no platform scraping — much simpler and more stable.

Run:
  pip install -r requirements.txt
  export GEMINI_API_KEY=...
  export GROQ_API_KEY=...
  uvicorn main:app --host 0.0.0.0 --port 8000
"""

import os
import json
import uuid
import shutil
import subprocess
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import requests

# ---------- setup ----------

BASE_DIR = Path(__file__).parent
WORK_DIR = BASE_DIR / "work"
CLIPS_DIR = BASE_DIR / "clips"
WORK_DIR.mkdir(exist_ok=True)
CLIPS_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Clip Finder")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to your frontend's domain in production
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/clips", StaticFiles(directory=str(CLIPS_DIR)), name="clips")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-flash-latest"

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_TRANSCRIBE_URL = "https://api.groq.com/openai/v1/audio/transcriptions"


MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB, adjust to your server's disk/bandwidth


class ClipResult(BaseModel):
    title: str
    start_seconds: float
    end_seconds: float
    reason: str
    download_url: str


# ---------- pipeline steps ----------

def save_upload(file: UploadFile, job_dir: Path) -> Path:
    """Stream the uploaded file to disk, enforcing a size cap as we go."""
    suffix = Path(file.filename or "source.mp4").suffix or ".mp4"
    out_path = job_dir / f"source{suffix}"
    written = 0
    with out_path.open("wb") as f:
        while chunk := file.file.read(1024 * 1024):
            written += len(chunk)
            if written > MAX_UPLOAD_BYTES:
                raise HTTPException(413, "File too large for this server's upload limit.")
            f.write(chunk)
    if written == 0:
        raise HTTPException(400, "Uploaded file is empty.")
    return out_path


def extract_audio(video_path: Path, job_dir: Path) -> Path:
    audio_path = job_dir / "audio.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(video_path), "-ac", "1", "-ar", "16000", str(audio_path)],
        check=True, capture_output=True,
    )
    return audio_path


def transcribe(audio_path: Path) -> str:
    """Returns a timestamped transcript string, e.g. '00:12 hello everyone\\n00:45 ...'.
    Uses Groq's hosted Whisper API so no model needs to be loaded into this server's own memory.
    Groq's free tier caps uploads at 25MB — long audio is split into 10-minute chunks first."""
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set on the server.")

    chunk_paths = _split_audio(audio_path)
    lines = []
    offset = 0.0
    for chunk_path in chunk_paths:
        with open(chunk_path, "rb") as f:
            resp = requests.post(
                GROQ_TRANSCRIBE_URL,
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                files={"file": (chunk_path.name, f, "audio/wav")},
                data={"model": "whisper-large-v3-turbo", "response_format": "verbose_json"},
                timeout=180,
            )
        resp.raise_for_status()
        data = resp.json()
        for seg in data.get("segments", []):
            start = seg["start"] + offset
            mm = int(start // 60)
            ss = int(start % 60)
            lines.append(f"{mm:02d}:{ss:02d} {seg['text'].strip()}")
        offset += _CHUNK_SECONDS
    return "\n".join(lines)


_CHUNK_SECONDS = 600  # 10 minutes per chunk, comfortably under Groq's 25MB free-tier limit


def _split_audio(audio_path: Path) -> list[Path]:
    """Split a wav file into _CHUNK_SECONDS-long pieces. Returns [audio_path] unchanged if it fits in one chunk."""
    duration = float(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(audio_path)],
        check=True, capture_output=True, text=True,
    ).stdout.strip())
    if duration <= _CHUNK_SECONDS:
        return [audio_path]

    chunks = []
    i = 0
    start = 0.0
    while start < duration:
        chunk_path = audio_path.parent / f"audio_chunk_{i}.wav"
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(audio_path), "-ss", str(start),
             "-t", str(_CHUNK_SECONDS), str(chunk_path)],
            check=True, capture_output=True,
        )
        chunks.append(chunk_path)
        start += _CHUNK_SECONDS
        i += 1
    return chunks


def find_highlights(transcript: str, max_clips: int) -> list[dict]:
    prompt = f"""You are an expert short-form video editor who finds viral-worthy clips from streamer/podcast VODs.
Below is a timestamped transcript. Find the {max_clips} MOST engaging, funny, shocking, or emotionally
interesting moments that would work as standalone 30-40 second short clips (TikTok/Reels/Shorts style).

Transcript:
\"\"\"
{transcript[:20000]}
\"\"\"

Respond ONLY with a JSON array, no markdown fences, no prose, like:
[{{"start_seconds": 45, "end_seconds": 82, "title": "short punchy hook title", "reason": "why this is clip-worthy in one sentence"}}]
Clips should be 25-45 seconds long. Order best first."""

    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set on the server.")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    resp = requests.post(
        url,
        headers={"x-goog-api-key": GEMINI_API_KEY, "Content-Type": "application/json"},
        json={
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"responseMimeType": "application/json"},
        },
        timeout=120,
    )
    resp.raise_for_status()
    text = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(text)


def cut_clip(video_path: Path, start: float, end: float, out_path: Path):
    duration = max(0.5, end - start)
    subprocess.run(
        [
            "ffmpeg", "-y", "-ss", str(start), "-i", str(video_path),
            "-t", str(duration), "-c:v", "libx264", "-c:a", "aac",
            "-preset", "fast", str(out_path),
        ],
        check=True, capture_output=True,
    )


# ---------- endpoint ----------

@app.post("/process", response_model=list[ClipResult])
def process(file: UploadFile = File(...), max_clips: int = Form(5)):
    job_id = uuid.uuid4().hex[:10]
    job_dir = WORK_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    try:
        video_path = save_upload(file, job_dir)
        audio_path = extract_audio(video_path, job_dir)
        transcript = transcribe(audio_path)
        if not transcript.strip():
            raise HTTPException(400, "Could not extract any speech from this video.")

        highlights = find_highlights(transcript, max_clips)

        results = []
        for i, clip in enumerate(highlights):
            clip_id = f"{job_id}_{i}.mp4"
            out_path = CLIPS_DIR / clip_id
            cut_clip(video_path, clip["start_seconds"], clip["end_seconds"], out_path)
            results.append(ClipResult(
                title=clip.get("title", f"Clip {i+1}"),
                start_seconds=clip["start_seconds"],
                end_seconds=clip["end_seconds"],
                reason=clip.get("reason", ""),
                download_url=f"/clips/{clip_id}",
            ))
        return results

    except subprocess.CalledProcessError as e:
        raise HTTPException(500, f"ffmpeg/yt-dlp error: {e.stderr[:500] if e.stderr else str(e)}")
    except Exception as e:
        raise HTTPException(500, str(e))
    finally:
        shutil.rmtree(job_dir, ignore_errors=True)  # keep only /clips, drop the raw download + audio


@app.get("/health")
def health():
    return {"status": "ok"}
