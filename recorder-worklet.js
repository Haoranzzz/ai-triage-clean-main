class PCMWorkletProcessor extends AudioWorkletProcessor {
    constructor() {
      super();
      this.inSR = sampleRate;          // 48000
      this.outSR = 16000;              // 16000
      this.buf48k = new Float32Array(0); // 仅存48k域
      this.IN_FRAME = 960;             // 20ms @48k
      this.OUT_FRAME = 320;            // 20ms @16k
      this.port.postMessage({type:'init', sr:this.inSR});
    }
  
    // 把 960 个 48k 样本线性重采样到 320 个 16k 样本
    _resample20ms_48k_to_16k(block48) {
      const out = new Float32Array(this.OUT_FRAME);
      const ratio = this.inSR / this.outSR; // 3.0
      for (let i = 0; i < this.OUT_FRAME; i++) {
        const idx = i * ratio;
        const i0 = Math.floor(idx);
        const frac = idx - i0;
        const s0 = block48[i0] || 0, s1 = block48[i0+1] || s0;
        out[i] = s0 + (s1 - s0) * frac;
      }
      return out;
    }
  
    process(inputs) {
      const chs = inputs[0];
      if (!chs || chs.length === 0) return true;
      const mono48 = chs[0];
      if (!mono48 || mono48.length === 0) return true;
  
      // 48k域拼接
      const joined = new Float32Array(this.buf48k.length + mono48.length);
      joined.set(this.buf48k, 0);
      joined.set(mono48, this.buf48k.length);
  
      // 一次可以切出多少个20ms@48k的块
      const blocks = Math.floor(joined.length / this.IN_FRAME);
  
      for (let b = 0; b < blocks; b++) {
        const start = b * this.IN_FRAME;
        const end   = start + this.IN_FRAME;
        const blk48 = joined.subarray(start, end);           // 960 @48k
        const f16   = this._resample20ms_48k_to_16k(blk48);  // 320 @16k
  
        // float32 -> int16, little-endian
        const pcm = new Int16Array(f16.length);
        for (let i = 0; i < f16.length; i++) {
          let s = Math.max(-1, Math.min(1, f16[i]));
          pcm[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
        }
        this.port.postMessage(pcm.buffer, [pcm.buffer]);     // 发送一帧
      }
  
      // 剩余不足 960 的留到下一次（仍在48k域）
      const used = blocks * this.IN_FRAME;
      this.buf48k = joined.subarray(used);
  
      this._tick = (this._tick || 0) + blocks;
      if (this._tick >= 50) {
        this.port.postMessage({type:'tick', frames: this._tick});
        this._tick = 0;
      }
      return true;
    }
  }
  registerProcessor('pcm-worklet', PCMWorkletProcessor);
  