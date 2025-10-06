from __future__ import annotations
from typing import AsyncIterator, List, Dict
import os, asyncio
from openai import AsyncOpenAI

MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
_api = os.getenv("OPENAI_API_KEY")
aclient = AsyncOpenAI(api_key=_api)

SYSTEM_PROMPT = (
    "You are a concise English-speaking medical intake assistant. "
    "Ask targeted follow-ups in short sentences. Do not diagnose."
)

SENTENCE_COMPLETION_PROMPT = (
    "Analyze if the following text represents a complete sentence or thought that the user intended to finish speaking. "
    "Consider context, grammar, and natural speech patterns. "
    "Respond with only 'COMPLETE' if it's a finished thought, or 'INCOMPLETE' if the user likely wants to continue speaking. "
    "Examples: 'Hello doctor' -> INCOMPLETE, 'I have a headache' -> COMPLETE, 'I think I might' -> INCOMPLETE, 'My symptoms started yesterday' -> COMPLETE"
)

async def stream_chat(messages: List[Dict[str, str]]) -> AsyncIterator[str]:
    # Chat Completions 流式，最稳
    stream = await aclient.chat.completions.create(
        model=MODEL,
        messages=[{"role":"system","content":SYSTEM_PROMPT}] + messages,
        stream=True,
        temperature=0.3,
    )
    async for chunk in stream:
        d = chunk.choices[0].delta
        if d and d.content:
            yield d.content

async def check_sentence_completion(text: str) -> bool:
    """
    使用ChatGPT判断句子是否完整
    返回True表示句子完整，可以停止录音
    返回False表示句子不完整，应该继续录音
    """
    if not text or len(text.strip()) < 3:
        return False
    
    try:
        response = await aclient.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SENTENCE_COMPLETION_PROMPT},
                {"role": "user", "content": f"Text to analyze: '{text.strip()}'"}
            ],
            temperature=0.1,
            max_tokens=10
        )
        
        result = response.choices[0].message.content.strip().upper()
        return result == "COMPLETE"
        
    except Exception as e:
        print(f"[LLM] Sentence completion check error: {e}")
        # 如果API调用失败，使用简单的启发式规则作为后备
        return _fallback_completion_check(text)

def _fallback_completion_check(text: str) -> bool:
    """
    后备的句子完整性检查（基于简单规则）
    """
    text = text.strip()
    if len(text) < 3:
        return False
    
    # 检查是否以句号、问号、感叹号结尾
    if text.endswith(('.', '?', '!')):
        return True
    
    # 检查是否包含完整的主谓结构（简单启发式）
    common_complete_patterns = [
        'i have', 'i feel', 'i am', 'i was', 'i need', 'i want',
        'my', 'the pain', 'it hurts', 'it started', 'it happens'
    ]
    
    text_lower = text.lower()
    for pattern in common_complete_patterns:
        if pattern in text_lower and len(text) > 10:
            return True
    
    return False
