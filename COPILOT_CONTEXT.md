# Project Context for Copilot

You are assisting in building **E-Hospital AI Assistant (Module 1)**.

## Mission
A doctor-facing AI gateway that handles **audio → transcription → reasoning → draft report → EHR write**, with the doctor always in the loop.

## Core Flow
1. Doctor speaks → audio uploaded by iOS frontend.  
2. FastAPI backend:
   - `/audio/transcribe`: sends audio to OpenAI (gpt-4o-mini-transcribe), returns TranscriptResponse.
   - `/llm/clarify`: returns clarifying questions if low-confidence.
   - `/llm/report`: returns structured SOAP/HPI JSON draft.
   - `/ehr/write`: writes approved notes into EHR.
   - `/patient/<id>/summary`: proxies EHR read.
3. Orchestrator: sequences the above steps.
4. Outputs: JSON packages (Transcript, Clarifier, Report, ActionPacket).

## Architecture
- **FastAPI backend**
  - `routes/` = API endpoints (thin).
  - `services/` = business logic (snapshot building, orchestration).
  - `clients/` = wrappers for external APIs (OpenAI, EHR).
  - `models/` = Pydantic schemas (Transcript, Snapshot, Report, etc).
- **Frontend**: Swift iOS app (not in this repo).
- **Inference layer**: OpenAI APIs (ASR, LLM).
- **Integration layer**: EHR API (read/write endpoints).

## Tech Preferences
- Python 3.11 + FastAPI.
- Pydantic for validation & schema export.
- httpx for HTTP calls.
- tenacity for retries.
- pytest for tests.
- .env for secrets (OPENAI_API_KEY, EHR base URL).
- Strict JSON schemas, always validate LLM outputs.
- Safety: never log PHI in plaintext. Audit actions instead.

## Coding Style
- Routes should only:
  - Parse input with Pydantic models.
  - Call service functions.
  - Return validated models.
- Services:
  - Contain business logic (filter/sort/join EHR data, orchestrate AI steps).
- Clients:
  - Contain external API calls (OpenAI, EHR).
  - Add retries, timeouts, and error handling.
- Always separate concerns clearly.

## Key Models
- TranscriptResponse: { kind: "transcript", data: Transcript }
- Transcript: { raw, cleaned, language, segments? }
- Snapshot: { patient_id, medical_history, medications, allergies, labs, diagnoses }
- Report: structured SOAP/HPI JSON.
- ActionPacket: scheduling/referral.

---

**When generating code:**
- Assume the repo structure exists.
- Use async where possible (httpx + FastAPI).
- Use clear logging without PHI.
- Always validate responses against Pydantic models.
