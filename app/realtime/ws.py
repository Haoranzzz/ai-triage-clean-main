"""
WebSocket router for real-time speech dialogue system.
Handles WebSocket connections, audio streaming, ASR, LLM, and TTS integration.
"""

from __future__ import annotations

import asyncio
import json
import uuid
import traceback
from typing import Callable, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from .asr import StreamingASR
from .llm import stream_chat
from .tts import stream_tts_chunks
from .state import SessionState

# Initialize global ASR instance for reuse
_asr = StreamingASR()

# Create FastAPI router
router = APIRouter()


@router.get("/ping")
async def ping():
    """Health check endpoint."""
    return {"ok": True}


class WebSocketHandler:
    """Handles WebSocket communication and message processing."""
    
    def __init__(self, websocket: WebSocket):
        self.ws = websocket
        self.state = SessionState(session_id=str(uuid.uuid4()))
        self.asr_task: Optional[asyncio.Task] = None
        self.llm_task: Optional[asyncio.Task] = None
        self.running = True
    
    async def safe_send(self, obj: dict) -> None:
        """Safely send JSON message to WebSocket client."""
        if self.ws.client_state != WebSocketState.CONNECTED:
            return
        try:
            await self.ws.send_json(obj)
        except Exception:
            pass
    
    def running_ref(self) -> bool:
        """Reference to running state for ASR task."""
        return self.running
    
    async def send_partial(self, text: str) -> None:
        """Send partial ASR result to client."""
        await self.safe_send({"type": "server.partial", "text": text})
    
    async def send_final(self, text: str) -> None:
        """Send final ASR result and trigger LLM response."""
        # Send final recognition result
        await self.safe_send({
            "type": "server.final", 
            "text": text, 
            "turn": self.state.turn
        })
        
        # Add user message to conversation history
        self.state.history.append({"role": "user", "content": text})
        
        # Cancel previous LLM task if still running
        if self.llm_task and not self.llm_task.done():
            self.llm_task.cancel()
        
        # Start new LLM task
        self.llm_task = asyncio.create_task(self._run_llm())
    
    async def _run_llm(self) -> None:
        """Run LLM inference and TTS generation."""
        response_buffer = []
        
        try:
            await self.safe_send({"type": "server.debug", "msg": "llm start"})
            
            # Stream LLM response
            async for token in stream_chat(self.state.history):
                response_buffer.append(token)
                await self.safe_send({"type": "server.token", "text": token})
            
            # Process complete response
            reply = "".join(response_buffer).strip()
            if reply:
                self.state.history.append({"role": "assistant", "content": reply})
            
            await self.safe_send({"type": "server.done", "turn": self.state.turn})
            await self.safe_send({
                "type": "server.debug", 
                "msg": f"llm done, {len(reply)} chars"
            })
            
            # Generate and stream TTS audio
            if reply:
                await self._stream_tts(reply)
                
        except Exception as e:
            traceback.print_exc()
            await self.safe_send({"type": "error", "msg": f"llm: {e}"})
    
    async def _stream_tts(self, text: str) -> None:
        """Stream TTS audio to client."""
        try:
            # Notify client that audio stream is starting
            await self.safe_send({
                "type": "server.say",
                "audio_seq": self.state.turn,
                "ended": False
            })
            
            # Stream audio chunks
            async for audio_chunk in stream_tts_chunks(text):
                await self.ws.send_bytes(audio_chunk)
            
            # Notify client that audio stream has ended
            await self.safe_send({
                "type": "server.say",
                "audio_seq": self.state.turn,
                "ended": True
            })
            
        except Exception as e:
            traceback.print_exc()
            await self.safe_send({"type": "error", "msg": f"tts: {e}"})
    
    async def handle_audio_start(self) -> None:
        """Handle audio recording start."""
        # Cancel previous LLM task
        if self.llm_task and not self.llm_task.done():
            self.llm_task.cancel()
        
        # Reset turn state
        self.state.reset_turn()
        self.running = True
        
        # Start ASR task
        self.asr_task = asyncio.create_task(
            _asr.run(
                audio_q=self.state.audio_q,
                send_partial=self.send_partial,
                send_final=self.send_final,
                running_ref=self.running_ref
            )
        )
        
        await self.safe_send({"type": "server.ack", "event": "audio.start"})
    
    async def handle_audio_stop(self) -> None:
        """Handle audio recording stop."""
        self.running = False
        
        try:
            # Wait for ASR task to complete with timeout
            if self.asr_task:
                await asyncio.wait_for(self.asr_task, timeout=2.0)
        except asyncio.TimeoutError:
            print("[WS] ASR task timeout, forcing completion")
            # Try to get accumulated text if available
            if hasattr(_asr, '_last_partial') and _asr._last_partial.strip():
                print(f"[WS] Sending accumulated text as final: '{_asr._last_partial}'")
                await self.send_final(_asr._last_partial)
        except Exception as e:
            print(f"[WS] ASR task error: {e}")
        
        await self.safe_send({"type": "server.ack", "event": "audio.stop"})
    
    async def handle_audio_pause(self) -> None:
        """Handle audio recording pause (for waiting AI response)."""
        # Temporarily pause recording but don't stop the session
        # This is used when waiting for AI response in continuous mode
        print("[WS] Audio recording paused for AI response")
        
        # Send acknowledgment to client
        await self.safe_send({"type": "server.ack", "event": "audio.pause"})
    
    async def handle_audio_data(self, audio_bytes: bytes) -> None:
        """Handle incoming audio data."""
        if self.running and self.asr_task is not None:
            await self.state.audio_q.put(audio_bytes)
            
            # Optional: Send reception acknowledgment for large data
            self.state._acc = getattr(self.state, "_acc", 0) + len(audio_bytes)
            if self.state._acc >= 65536:  # 64KB threshold
                await self.safe_send({
                    "type": "server.partial",
                    "text": f"rx {self.state._acc} bytes"
                })
                self.state._acc = 0
    
    async def handle_message(self, message: dict) -> None:
        """Handle incoming WebSocket message."""
        message_type = message.get("type")
        
        if message_type == "join":
            await self.safe_send({
                "type": "server.joined",
                "session_id": self.state.session_id
            })
        
        elif message_type == "audio.start":
            await self.handle_audio_start()
        
        elif message_type == "audio.stop":
            await self.handle_audio_stop()
        
        elif message_type == "audio.pause":
            await self.handle_audio_pause()
        
        else:
            await self.safe_send({
                "type": "error",
                "msg": f"unknown message type: {message_type}"
            })
    
    async def cleanup(self) -> None:
        """Clean up resources when connection closes."""
        self.running = False
        
        if self.asr_task:
            self.asr_task.cancel()
        
        if self.llm_task:
            self.llm_task.cancel()


@router.websocket("/ws/talk")
async def talk(websocket: WebSocket):
    """Main WebSocket endpoint for real-time speech dialogue."""
    await websocket.accept()
    
    handler = WebSocketHandler(websocket)
    
    try:
        while True:
            message = await websocket.receive()
            
            # Handle text messages (JSON)
            if message.get("text") is not None:
                try:
                    data = json.loads(message["text"])
                    await handler.handle_message(data)
                except json.JSONDecodeError:
                    await handler.safe_send({
                        "type": "error",
                        "msg": "Invalid JSON format"
                    })
                except Exception as e:
                    await handler.safe_send({
                        "type": "error",
                        "msg": f"Message handling error: {e}"
                    })
            
            # Handle binary messages (audio data)
            elif message.get("bytes") is not None:
                try:
                    await handler.handle_audio_data(message["bytes"])
                except Exception as e:
                    await handler.safe_send({
                        "type": "error",
                        "msg": f"Audio data handling error: {e}"
                    })
    
    except WebSocketDisconnect:
        print(f"[WS] Client disconnected: {handler.state.session_id}")
    
    except Exception as e:
        print(f"[WS] Unexpected error: {e}")
        traceback.print_exc()
    
    finally:
        await handler.cleanup()
