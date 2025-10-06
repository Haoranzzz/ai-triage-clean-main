from __future__ import annotations
import asyncio
from dataclasses import dataclass, field
from typing import Optional,List, Dict

@dataclass
class SessionState:
    session_id: str
    audio_q: asyncio.Queue = field(default_factory=asyncio.Queue)  # bytes of PCM16 mono 16k
    running: bool = False
    turn: int = 0
    history: List[Dict] = field(default_factory=list)  # [{"role":"user"/"assistant","content":"..."}]

    def reset_turn(self):
        self.turn += 1
