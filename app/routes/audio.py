from fastapi import APIRouter, UploadFile, File, HTTPException
from app.clients.openai_client import OpenAIClient
from app.models.transcript import TranscriptResponse
import tempfile
import shutil
import os

router = APIRouter()
client = OpenAIClient()

"""
    Accept an audio file upload, send it to OpenAI ASR,
    and return a validated TranscriptResponse.
    """

@router.post("/transcribe", response_model=TranscriptResponse)
async def transcribe_audio(file: UploadFile = File(...)):
    #save the uploaded file to a temporary location
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as temp:
            shutil.copyfileobj(file.file, temp)
            temp_path = temp.name
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving temp file: {e}")
    
    #run transcription
    try: 
        result = client.transcribe_audio(temp_path)
    except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error during transcription: {e}")


    return result

        