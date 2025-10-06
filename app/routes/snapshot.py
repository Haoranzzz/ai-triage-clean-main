# app/routes/snapshot.py
from typing import Dict
import os
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from app.services.snapshot_builder import build_snapshot

router = APIRouter()

# In-memory consent store
CONSENT: Dict[int, bool] = {}

@router.get("/consent")
def set_consent(patient_id: int, granted: bool):
    try:
        CONSENT[patient_id] = bool(granted)
        return {
            "patient_id": patient_id,
            "consent_granted": CONSENT[patient_id],
            "message": "Consent recorded"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/snapshot")
async def snapshot(patient_id: int = Query(..., ge=1)):
    if not CONSENT.get(patient_id, False):
        raise HTTPException(status_code=403, detail="Consent required or denied for this patient_id")
    try:
        return await build_snapshot(patient_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class ChatIn(BaseModel):
    patient_id: int
    prompt: str

@router.post("/chat")
async def chat(body: ChatIn):
    if not CONSENT.get(body.patient_id, False):
        raise HTTPException(status_code=403, detail="Consent required or denied for this patient_id")

    use_openai = os.getenv("USE_OPENAI", "false").lower() == "true"
    api_key = os.getenv("OPENAI_API_KEY", "").strip()

    snap = await build_snapshot(body.patient_id)

    # MOCK MODE
    if (not use_openai) or (not api_key):
        mh = snap.get("medical_history", [])
        dx = snap.get("diagnoses", [])
        meds = snap.get("medications", [])
        labs = snap.get("labs", [])

        latest_dx = dx[0] if dx else None
        latest_mh = mh[0] if mh else None
        latest_med = meds[0] if meds else None
        latest_lab = labs[0] if labs else None

        answer = {
            "summary": {
                "patient_id": body.patient_id,
                "recent_condition": latest_mh.get("condition") if latest_mh else None,
                "recent_diagnosis": latest_dx.get("diagnosis_description") if latest_dx else None,
                "recent_medication": latest_med.get("medicine_name") if latest_med else None,
                "recent_lab": {
                    "test_type": latest_lab.get("test_type") if latest_lab else None,
                    "result": latest_lab.get("result") if latest_lab else None,
                    "date": latest_lab.get("test_date") if latest_lab else None,
                } if latest_lab else None
            },
            "clarifying_questions": [
                "When did the current symptoms start?",
                "Any allergies or recent medication changes?"
            ],
            "next_steps": [
                "Review vitals and recent labs",
                "Schedule follow-up if symptoms persist or worsen"
            ]
        }
        return {"mode": "mock", "answer": answer, "snapshot": snap}

    # REAL MODE: OpenAI
    try:
        from openai import OpenAI
        client = OpenAI
        prompt = (
            "You are a clinical assistant. Use ONLY the provided patient snapshot to answer. "
            "If data is missing, say so and propose safe next steps.\n\n"
            f"User prompt: {body.prompt}\n\n"
            f"Patient snapshot (JSON): {snap}\n"
        )
        resp = client.responses.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            input=prompt,
        )
        return {"mode": "openai", "answer": getattr(resp, "output_text", None), "raw": resp}
    except Exception as e:
        return {"mode": "error-fallback-mock", "error": str(e), "snapshot": snap}

@router.post("/chatmock")
async def chatmock(body: ChatIn):
    if not CONSENT.get(body.patient_id, False):
        raise HTTPException(status_code=403, detail="Consent required or denied for this patient_id")

    snap = await build_snapshot(body.patient_id)

    mh = snap.get("medical_history", [])
    dx = snap.get("diagnoses", [])
    meds = snap.get("medications", [])
    labs = snap.get("labs", [])

    latest_dx = dx[0] if dx else None
    latest_mh = mh[0] if mh else None
    latest_med = meds[0] if meds else None
    latest_lab = labs[0] if labs else None

    answer = {
        "summary": {
            "patient_id": body.patient_id,
            "recent_condition": latest_mh.get("condition") if latest_mh else None,
            "recent_diagnosis": latest_dx.get("diagnosis_description") if latest_dx else None,
            "recent_medication": latest_med.get("medicine_name") if latest_med else None,
            "recent_lab": {
                "test_type": latest_lab.get("test_type") if latest_lab else None,
                "result": latest_lab.get("result") if latest_lab else None,
                "date": latest_lab.get("test_date") if latest_lab else None,
            } if latest_lab else None
        },
        "clarifying_questions": [
            "When did the current symptoms start?",
            "Any allergies or recent medication changes?"
        ],
        "next_steps": [
            "Review vitals and recent labs",
            "Schedule follow-up if symptoms persist or worsen"
        ]
    }
    return {"mode": "mock-direct", "answer": answer, "snapshot": snap}