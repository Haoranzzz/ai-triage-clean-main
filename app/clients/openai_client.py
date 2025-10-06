# app/clients/openai_client.py
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from openai import OpenAI
from app.models.transcript import Transcript, TranscriptResponse

# Load environment
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set in .env")

client = OpenAI(api_key=OPENAI_API_KEY)

class OpenAIClient:
    def __init__(self, model: str = "gpt-4o-mini-transcribe"):
        self.model = model

    def transcribe_audio(self, file_path: str, language: Optional[str] = "en") -> TranscriptResponse:
        path = Path(file_path)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        with open(path, "rb") as f:
            resp = client.audio.transcriptions.create(
                model=self.model,
                file=f
            )

        text = resp.text.strip()

        transcript = Transcript(
            raw=resp.text,
            cleaned=text,
            language=language or "en"
        )
        return TranscriptResponse(kind="transcript", data=transcript)
