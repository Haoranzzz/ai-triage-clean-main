from __future__ import annotations
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from .llm import stream_chat
from starlette.websockets import WebSocketState

import asyncio, json, uuid

from .state import SessionState
#from .asr import StreamingASR, FRAME_MS

#from app.realtime.asr import RealtimeASR
#_asr = RealtimeASR()
from app.realtime.asr import StreamingASR
_asr = StreamingASR()
router = APIRouter()
#_asr = StreamingASR(model_size="tiny.en")  # 全局复用模型

@router.get("/ping")
async def ping():
    return {"ok": True}

@router.websocket("/ws/talk")
async def talk(ws: WebSocket):
    await ws.accept()
    async def safe_send(obj):
        if ws.client_state != WebSocketState.CONNECTED:
            return
        try:
            await ws.send_json(obj)
        except Exception:
            pass

    state = SessionState(session_id=str(uuid.uuid4()))
    asr_task: asyncio.Task | None = None
    llm_task: asyncio.Task | None = None
    running = True

    def running_ref(): return running

    async def send_partial(text: str):
        await safe_send({"type": "server.partial", "text": text})

    async def send_final(text: str):
        nonlocal llm_task
        # 1) 回显最终识别
        await safe_send({"type": "server.final", "text": text, "turn": state.turn})
        state.history.append({"role": "user", "content": text})

        # 2) 打断上一次还在跑的 LLM
        if llm_task and not llm_task.done():
            llm_task.cancel()

        async def _run_llm():
            buf = []
            try:
                await safe_send({"type": "server.debug", "msg": "llm start"})
                async for tok in stream_chat(state.history):
                    buf.append(tok)
                    await safe_send({"type": "server.token", "text": tok})
                reply = "".join(buf).strip()
                if reply:
                    state.history.append({"role": "assistant", "content": reply})
                await safe_send({"type": "server.done", "turn": state.turn})
                await safe_send({"type": "server.debug", "msg": f"llm done, {len(reply)} chars"})
                
                # TTS播放
                if reply:
                    # 下行前导，通知前端即将收到二进制音频
                    await safe_send({"type":"server.say","audio_seq":state.turn,"ended":False})
                    async for b in stream_tts_chunks(reply):
                        await ws.send_bytes(b)
                    await safe_send({"type":"server.say","audio_seq":state.turn,"ended":True})
                    
            except Exception as e:
                # 同时向控制台与客户端报告
                import traceback; traceback.print_exc()
                await safe_send({"type":"error","msg":f"llm:{e}"})

        llm_task = asyncio.create_task(_run_llm())

    try:
        while True:
            msg = await ws.receive()

            if msg.get("text") is not None:
                try:
                    data = json.loads(msg["text"])
                except Exception:
                    await safe_send({"type":"error","msg":"bad json"}); continue
                t = data.get("type")

                if t == "join":
                    await safe_send({"type":"server.joined","session_id":state.session_id})

                elif t == "audio.start":
                    # 打断上一次的 LLM 回复
                    if llm_task and not llm_task.done():
                        llm_task.cancel()
                    state.reset_turn()
                    running = True
                    asr_task = asyncio.create_task(
                        _asr.run(
                            audio_q=state.audio_q,
                            send_partial=send_partial,
                            send_final=send_final,
                            running_ref=running_ref
                        )
                    )
                    await safe_send({"type":"server.ack","event":"audio.start"})

                elif t == "audio.stop":
                    running = False
                    try:
                        # 等待ASR任务完成，确保发送最终结果
                        if asr_task: 
                            await asyncio.wait_for(asr_task, timeout=2.0)  # 增加超时时间
                    except asyncio.TimeoutError:
                        print("[WS] ASR task timeout, forcing completion")
                        # 如果ASR任务超时，尝试强制获取当前累积的文本
                        if hasattr(_asr, '_last_partial') and _asr._last_partial.strip():
                            print(f"[WS] Sending accumulated text as final: '{_asr._last_partial}'")
                            await send_final(_asr._last_partial)
                    except Exception as e:
                        print(f"[WS] ASR task error: {e}")
                    await safe_send({"type":"server.ack","event":"audio.stop"})

                else:
                    await safe_send({"type":"error","msg":f"unknown:{t}"})

            elif msg.get("bytes") is not None:
                if running and asr_task is not None:
                    b = msg["bytes"]
                    await state.audio_q.put(b)
                    # 可选的收包回执，便于前端观察
                    state._acc = getattr(state, "_acc", 0) + len(b)
                    if state._acc >= 65536:
                        await safe_send({"type":"server.partial","text":f"rx {state._acc} bytes"})
                        state._acc = 0

    except WebSocketDisconnect:
        running = False
        if asr_task: asr_task.cancel()
        if llm_task: llm_task.cancel()
        return

# 添加缺失的TTS导入
from .tts import stream_tts_chunks

import asyncio
import base64
import json
import os
import sounddevice as sd
import numpy as np
import websockets
from dotenv import load_dotenv
import requests

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

# 生成临时会话（Realtime）
def create_realtime_session():
    r = requests.post(
        "https://api.openai.com/v1/realtime/sessions",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "gpt-4o-realtime-preview",
            "voice": "alloy",
            "turn_detection": {"type": "server_vad"}
        }
    )
    r.raise_for_status()
    return r.json()

# 麦克风参数
RATE = 16000
BATCH_MS = 150                        # 减少到150ms，提高响应速度
BATCH_SAMPLES = RATE * BATCH_MS // 1000

audio_buffer = np.zeros(0, dtype=np.int16)

async def realtime_conversation():
    # 创建 Realtime 临时会话
    sess = create_realtime_session()
    ws_url = sess["client_secret"]["value"]
    model_endpoint = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview"

    async with websockets.connect(
        model_endpoint,
        additional_headers={
            "Authorization": f"Bearer {sess['client_secret']['value']}",
            "OpenAI-Beta": "realtime=v1"
        },
        subprotocols=["realtime"],   # 建议加入
        compression=None   
    ) as ws:
        print("Connected to ChatGPT Realtime.")
        audio_buffer = np.zeros(0, dtype=np.int16)

        def callback(indata, frames, time, status):
            nonlocal audio_buffer
            if status: print(status)
            audio_buffer = np.append(audio_buffer, indata.copy())

        # 开启麦克风录音
        with sd.InputStream(
            samplerate=RATE, channels=1, dtype="int16", callback=callback
        ):
            try:
                while True:
                    await asyncio.sleep(0.1)  # 减少延迟从0.5秒到0.1秒
                    if len(audio_buffer) >= BATCH_SAMPLES:
                        continue
                    # 把音频编码成 base64，发送到服务器端
                    pcm_bytes = audio_buffer.tobytes()
                    b64_audio = base64.b64encode(pcm_bytes).decode("utf-8")
                    await ws.send(json.dumps({
                        "type": "input_audio_buffer.append",
                        "audio": b64_audio
                    }))
                    await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
                    await ws.send(json.dumps({
                        "type": "response.create",
                        "response": {"modalities": ["audio","text"]}
                    }))
                    audio_buffer = np.zeros(0, dtype=np.int16)
                    # 监听返回音频
                    msg = await ws.recv()
                    if "audio" in msg:
                        print("Response:", msg)
            except KeyboardInterrupt:
                print("Stopped.")

if __name__ == "__main__":
    asyncio.run(realtime_conversation())
