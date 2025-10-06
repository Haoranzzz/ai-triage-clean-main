'''
from __future__ import annotations
import asyncio
import numpy as np
import webrtcvad
from faster_whisper import WhisperModel
from typing import Callable, Optional

PCM_BYTES_PER_MS = 16_000 * 2 // 1000  # mono 16k, s16le => 32 bytes per ms
FRAME_MS = 20                           # client sends 20ms frames
SIL_MS = 600                            # endpoint: >=600ms silence
MIN_UTT_MS = 400                        # minimal voiced length to decode

class StreamingASR:
    def __init__(self, model_size: str = "tiny", cpu_threads: int = 4):
        # CPU 默认即可；后续可改 device="cuda"
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8", cpu_threads=cpu_threads)
        self.vad = webrtcvad.Vad(2)  # 0..3，越大越严格
        self.voice_buf = bytearray()
        self.sil_acc_ms = 0
        self.has_voice = False

    def _bytes_to_float32(self, b: bytes) -> np.ndarray:
        # int16 -> float32 [-1,1]
        arr = np.frombuffer(b, dtype=np.int16).astype(np.float32) / 32768.0
        return arr

    async def run(
        self,
        audio_q: asyncio.Queue,
        send_partial: Callable[[str], asyncio.Future],
        send_final:   Callable[[str], asyncio.Future],
        running_ref: Callable[[], bool]
    ):
        """
        拉取 20ms PCM 帧，VAD 切段；静音达到阈值就 decode 一段。
        """
        # 为 partial 累加一个滚动窗，避免太短
        partial_acc = bytearray()
        while running_ref():
            try:
                frame: bytes = await asyncio.wait_for(audio_q.get(), timeout=0.5)
            except asyncio.TimeoutError:
                # 长时间没帧，当作静音推进
                if self.has_voice and self.sil_acc_ms >= SIL_MS and len(self.voice_buf) >= PCM_BYTES_PER_MS * MIN_UTT_MS:
                    await self._flush_segment(send_final)
                continue

            # VAD 需要 10/20/30ms 帧；我们就是 20ms
            voiced = self.vad.is_speech(frame, sample_rate=16_000)

            if voiced:
                self.has_voice = True
                self.sil_acc_ms = 0
                self.voice_buf += frame
                partial_acc += frame
                # 每 ~800ms 给一次 partial
                if len(partial_acc) >= PCM_BYTES_PER_MS * 800:
                    txt = await self._decode(partial_acc, beam_size=1)
                    if txt.strip():
                        await send_partial(txt)
                    partial_acc.clear()
            else:
                if self.has_voice:
                    self.sil_acc_ms += FRAME_MS
                    self.voice_buf += frame  # 保留少量尾部静音更稳

                # endpoint
                if self.has_voice and self.sil_acc_ms >= SIL_MS:
                    await self._flush_segment(send_final)
                    partial_acc.clear()

    async def _flush_segment(self, send_final):
        if len(self.voice_buf) < PCM_BYTES_PER_MS * MIN_UTT_MS:
            self.voice_buf.clear()
            self.sil_acc_ms = 0
            self.has_voice = False
            return
        txt = await self._decode(self.voice_buf, beam_size=2)
        self.voice_buf.clear()
        self.sil_acc_ms = 0
        self.has_voice = False
        if txt.strip():
            await send_final(txt)

    async def _decode(self, pcm_bytes: bytes, beam_size: int = 1) -> str:
        audio = self._bytes_to_float32(pcm_bytes)
        segments, _ = self.model.transcribe(audio, language="en", task="transcribe", beam_size=beam_size, vad_filter=False)
        out = []
        for s in segments:
            out.append(s.text)
        return "".join(out).strip()
        '''

# app/realtime/asr.py
from __future__ import annotations
import os, io, struct, asyncio, time
from typing import Callable
from openai import AsyncOpenAI
from .llm import check_sentence_completion

# 输入约定：前端发送 16kHz / 16-bit / mono / little-endian PCM，20ms 一帧（640字节）
SAMPLE_RATE = 16_000
FRAME_MS    = 20
BYTES_PER_MS = (SAMPLE_RATE * 2) // 1000  # 32 bytes/ms
CHUNK_MS    = 600                         # 减少到 ~0.6s 做一次"伪流式"转写，提高响应速度
ASR_MODEL   = os.getenv("OPENAI_ASR_MODEL", "gpt-4o-mini-transcribe")

_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class StreamingASR:
    """
    智能流式ASR：
      - 按 ~0.8s 聚合一段 PCM
      - 转成内存 WAV
      - 调 OpenAI Transcriptions 得到文本
      - 使用ChatGPT判断句子完整性，智能决定是否停止录音
      - 聚合期间给 partial，检测到完整句子后给 final
    """

    def __init__(self):
        self.running_ref = None
        self.pcm_buffer = bytearray()
        self.accumulated_text = ""
        self.silence_count = 0
        self._completion_check_count = 0
        self._max_completion_checks = 3  # 最多检查3次
        self._final_sent = False
        self._last_partial = ""  # 添加这个属性以便外部访问

    async def run(
        self,
        audio_q: asyncio.Queue,
        send_partial: Callable[[str], asyncio.Future],
        send_final:   Callable[[str], asyncio.Future],
        running_ref:  Callable[[], bool],
    ):
        # 重置状态
        self.running_ref = running_ref
        self._buf = bytearray()
        self._last_flush_ms = self._now_ms()
        self._last_partial = ""
        self._final_sent = False
        self._completion_check_count = 0
        
        silence_count = 0
        max_silence_chunks = 12  # 增加到12 * 0.8s = 9.6s 静音后自动停止
        accumulated_sentences = []  # 累积多个句子
        current_sentence = ""
        
        # 主循环：收帧 → 累积 → 定时转写（partial）→ 智能完整性检测
        while True:
            # 检查用户是否手动停止
            if not running_ref():
                print("[ASR] User manually stopped recording")
                break

            # 检查音频队列是否为空（超时处理）
            if audio_q.empty():
                await asyncio.sleep(0.05)  # 减少延迟从0.1秒到0.05秒
                silence_count += 1
                # 如果长时间没有音频输入且有内容，自动结束
                if silence_count > max_silence_chunks and len(self._buf) > 0:
                    print("[ASR] Long silence detected, auto stopping")
                    break
                continue
            else:
                silence_count = 0  # 重置静音计数

            try:
                frame: bytes = await asyncio.wait_for(audio_q.get(), timeout=0.5)
                self._buf += frame
            except asyncio.TimeoutError:
                continue

            # 定时做一次 partial 和智能完整性检测
            if self._now_ms() - self._last_flush_ms >= CHUNK_MS and len(self._buf) >= BYTES_PER_MS * CHUNK_MS:
                txt = await self._decode(bytes(self._buf))
                if txt:
                    # 检查是否是新的句子内容
                    if txt != current_sentence:
                        current_sentence = txt
                        
                        # 只把新增的差量发出去，避免 "Hello." 重复
                        delta = txt[len(self._last_partial):]
                        if delta.strip():
                            await send_partial(delta)
                            self._last_partial = txt  # 保存完整文本供外部访问
                    
                    # 检查是否包含多个句子（通过句号、问号、感叹号分割）
                    sentences = self._split_into_sentences(txt)
                    
                    # 如果检测到完整句子，但不立即停止录音，而是继续收集
                    if len(sentences) > 1 or (len(sentences) == 1 and self._is_sentence_complete(sentences[0])):
                        # 将完整的句子添加到累积列表
                        for sentence in sentences[:-1]:  # 除了最后一个句子
                            if sentence.strip() and sentence.strip() not in accumulated_sentences:
                                accumulated_sentences.append(sentence.strip())
                        
                        # 检查最后一个句子是否完整
                        last_sentence = sentences[-1] if sentences else ""
                        if last_sentence.strip() and self._is_sentence_complete(last_sentence):
                            accumulated_sentences.append(last_sentence.strip())
                        
                        print(f"[ASR] Accumulated sentences: {accumulated_sentences}")
                        
                        # 只有在检测到明确的停顿或特定条件时才停止
                        if (self._completion_check_count >= 2 and  # 至少检查2次
                            len(accumulated_sentences) > 0 and
                            silence_count > 3):  # 有一定的静音
                            
                            # 合并所有句子作为最终结果
                            final_text = " ".join(accumulated_sentences)
                            if last_sentence.strip() and not self._is_sentence_complete(last_sentence):
                                final_text += " " + last_sentence.strip()
                            
                            print(f"[ASR] Sending combined sentences: '{final_text}'")
                            await send_final(final_text)
                            self._final_sent = True
                            break
                    
                    self._completion_check_count += 1
                    
                self._last_flush_ms = self._now_ms()

        # 结束：给 final（用整段，包括所有累积的句子）
        if self._buf and not self._final_sent:
            txt = await self._decode(bytes(self._buf))
            if txt:
                # 如果有累积的句子，合并它们
                if accumulated_sentences:
                    sentences = self._split_into_sentences(txt)
                    for sentence in sentences:
                        if sentence.strip() and sentence.strip() not in accumulated_sentences:
                            accumulated_sentences.append(sentence.strip())
                    final_text = " ".join(accumulated_sentences)
                else:
                    final_text = txt
                
                self._last_partial = final_text  # 确保最终文本也被保存
                await send_final(final_text)
                self._final_sent = True
        self._buf.clear()
        print(f"[ASR] Run completed, final text: '{self._last_partial}'")

    def _split_into_sentences(self, text: str) -> list:
        """将文本分割成句子"""
        import re
        # 使用正则表达式分割句子，保留标点符号
        sentences = re.split(r'([.!?。！？]+)', text)
        result = []
        for i in range(0, len(sentences) - 1, 2):
            sentence = sentences[i]
            if i + 1 < len(sentences):
                sentence += sentences[i + 1]  # 添加标点符号
            if sentence.strip():
                result.append(sentence.strip())
        
        # 如果最后一个部分没有标点符号，也添加进去
        if len(sentences) % 2 == 1 and sentences[-1].strip():
            result.append(sentences[-1].strip())
        
        return result if result else [text.strip()]

    def _is_sentence_complete(self, sentence: str) -> bool:
        """简单判断句子是否完整（基于标点符号）"""
        sentence = sentence.strip()
        return (sentence.endswith(('.', '!', '?', '。', '！', '？')) and 
                len(sentence) > 5)  # 至少5个字符才认为是完整句子

    async def _should_stop_recording(self, text: str) -> bool:
        """
        使用ChatGPT判断是否应该停止录音
        """
        try:
            # 检查句子完整性
            is_complete = await check_sentence_completion(text)
            
            # 更严格的完整性判断：需要同时满足多个条件
            if is_complete:
                # 检查是否包含常见的句子结束标志
                has_ending_punctuation = any(text.strip().endswith(p) for p in ['.', '!', '?', '。', '！', '？'])
                
                # 检查长度是否合理（避免过短的句子被误判）
                reasonable_length = len(text.strip()) >= 15
                
                # 检查是否不是明显的不完整句子开头
                not_incomplete_start = not any(text.strip().lower().startswith(start) for start in [
                    'i am', 'i want', 'i need', 'i think', 'i would', 'i have',
                    'can you', 'could you', 'would you', 'do you', 'are you',
                    'what', 'where', 'when', 'why', 'how', 'who'
                ])
                
                if has_ending_punctuation and reasonable_length:
                    print(f"[ASR] Complete sentence detected with punctuation: '{text}'")
                    return True
                elif reasonable_length and not_incomplete_start and len(text.strip()) > 30:
                    print(f"[ASR] Likely complete sentence (no punctuation but reasonable): '{text}'")
                    return True
            
            # 如果文本很长（超过120字符），即使不完全确定也倾向于停止，避免录音过长
            if len(text.strip()) > 120:
                print(f"[ASR] Text too long ({len(text)} chars), forcing stop")
                return True
                
            return False
            
        except Exception as e:
            print(f"[ASR] Error in completion check: {e}")
            # 如果检测失败，使用更保守的长度策略
            return len(text.strip()) > 80

    def _calculate_audio_energy(self, pcm_bytes: bytes) -> float:
        """计算音频能量，用于判断是否有实际语音内容"""
        import numpy as np
        if not pcm_bytes:
            return 0.0
        
        # 将PCM字节转换为numpy数组
        audio_data = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
        # 计算RMS能量
        rms = np.sqrt(np.mean(audio_data ** 2))
        return rms

    async def _decode(self, pcm_bytes: bytes) -> str:
        """
        把 16kHz s16le PCM 包成 WAV 后，调用 OpenAI 转写。
        增加音频质量检测，避免误识别静音或噪音。
        """
        if not pcm_bytes:
            return ""
        
        # 确保有足够的音频数据进行转写
        min_audio_length = SAMPLE_RATE * 0.5  # 增加到至少0.5秒的音频
        if len(pcm_bytes) < min_audio_length * 2:  # *2 因为是16位音频
            print(f"[ASR] Audio too short ({len(pcm_bytes)} bytes), skipping transcription")
            return ""
        
        # 计算音频能量，过滤掉静音或极低音量的音频
        audio_energy = self._calculate_audio_energy(pcm_bytes)
        min_energy_threshold = 100.0  # 最小能量阈值，可根据实际情况调整
        
        if audio_energy < min_energy_threshold:
            print(f"[ASR] Audio energy too low ({audio_energy:.2f}), likely silence, skipping transcription")
            return ""
            
        wav_bytes = self._pcm_to_wav(pcm_bytes)
        bio = io.BytesIO(wav_bytes)
        bio.name = "chunk.wav"  # SDK 需要一个文件名
        try:
            resp = await _client.audio.transcriptions.create(
                model=ASR_MODEL,
                file=bio,
                language="en",
                response_format="text",  # 使用text格式，兼容性更好
                temperature=0.0,  # 降低随机性，提高一致性
            )
            
            # 直接获取文本结果
            text = (resp or "").strip()
            
            # 过滤掉常见的误识别结果
            false_positive_patterns = [
                "i am", "i'm", "um", "uh", "ah", "oh", "mm", "hmm",
                ".", ",", "?", "!", " ", ""
            ]
            
            # 检查是否为误识别的常见模式
            if text.lower().strip() in false_positive_patterns:
                print(f"[ASR] Filtered out likely false positive: '{text}'")
                return ""
            
            # 检查文本长度和内容质量
            if len(text.strip()) < 3:  # 过短的结果很可能是误识别
                print(f"[ASR] Text too short, likely false positive: '{text}'")
                return ""
            
            # 记录转写结果用于调试
            if text:
                print(f"[ASR] Transcribed {len(pcm_bytes)} bytes (energy: {audio_energy:.2f}) -> '{text}'")
            else:
                print(f"[ASR] No transcription result for {len(pcm_bytes)} bytes")
                
            return text
            
        except Exception as e:
            # 控制台打印即可；不中断主流程
            print(f"[ASR] Transcription error: {repr(e)}")
            return ""

    def _pcm_to_wav(self, pcm: bytes) -> bytes:
        """把裸 PCM(16kHz, s16le, mono) 封装成最小 WAV。"""
        num_channels = 1
        bits_per_sample = 16
        byte_rate  = SAMPLE_RATE * num_channels * bits_per_sample // 8
        block_align = num_channels * bits_per_sample // 8
        data_size = len(pcm)
        header = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF", 36 + data_size, b"WAVE",
            b"fmt ", 16, 1, num_channels,
            SAMPLE_RATE, byte_rate, block_align, bits_per_sample,
            b"data", data_size
        )
        return header + pcm

    @staticmethod
    def _now_ms() -> int:
        return int(time.time() * 1000)
