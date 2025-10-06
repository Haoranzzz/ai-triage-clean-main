
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware


from app.routes import snapshot
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])

@app.get("/")
def home():
    return {"message": "API is running. Try /health or /docs"}

@app.get("/health")
def health():
    return {"ok": True}

# Mount app routes
app.include_router(snapshot.router,prefix="/EHR", tags=["app"])

# app/routes/audio.py
from app.routes import audio
app.include_router(audio.router, prefix="/audio", tags=["audio"])

app.include_router(snapshot.router,prefix="/EHR", tags=["app"])

from app.routes import reasoning
# Mount reasoning router
app.include_router(reasoning.router, prefix="/reas", tags=["reasoning"])


from app.realtime.ws import router as realtime_router
app.include_router(realtime_router, prefix="/realtime", tags=["realtime"])

@app.get("/health")
def health(): return {"ok": True}