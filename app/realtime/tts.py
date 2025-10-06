from __future__ import annotations
from typing import AsyncIterator
import os, aiohttp

# 示例：OpenAI TTS 流 to PCM/WAV（简化，后续可换）
async def stream_tts_chunks(text: str) -> AsyncIterator[bytes]:
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    # 返回 mp3 流也行，但前端解码麻烦。先用 wav。
    async with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="alloy",
        input=text,
        response_format="wav"
    ) as resp:
        async for chunk in resp.iter_bytes(8192):
            yield chunk
