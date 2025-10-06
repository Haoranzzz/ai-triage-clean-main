from pydantic import BaseModel

class Transcript(BaseModel):
    """
    Represents a single transcript object.
    - raw: exact text returned by OpenAI ASR
    - cleaned: normalized text (remove fillers, fix grammar, etc.)
    - language: detected or provided language code
    """
    raw: str
    cleaned: str
    language: str

class TranscriptResponse(BaseModel):
    """
    Represents the full response from the transcription endpoint.
    - transcript: the Transcript object containing raw and cleaned text
    """
    kind : str
    data : Transcript
    