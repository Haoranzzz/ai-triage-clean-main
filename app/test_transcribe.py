import os
from dotenv import load_dotenv
from openai import OpenAI

# Load .env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
print("API key loaded?", api_key[:10] + "...")

client = OpenAI(api_key=api_key)

# Try a transcription call
with open("Rideau St.m4a", "rb") as f:
    resp = client.audio.transcriptions.create(
        model="gpt-4o-mini-transcribe",
        file=f
    )

print("Transcription:", resp.text[:200])
