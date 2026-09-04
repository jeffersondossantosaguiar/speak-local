import { useCallback, useEffect, useRef, useState } from "react";

type Status = "idle" | "recording" | "error";

interface RecorderState {
  status: Status;
  error: string | null;
  recordingSeconds: number;
  start: () => Promise<void>;
  stop: () => Promise<Blob | null>;
}

/**
 * Records audio via MediaRecorder (produces WebM/Opus on Chromium/Firefox).
 * The browser decodes the compressed blob to PCM via WebAudio, re-encodes it as
 * an uncompressed WAV, and that WAV is what gets uploaded. This avoids needing
 * an Opus decoder on the Rust side (Symphonia does not ship one).
 */
export function useMediaRecorder(): RecorderState {
  const [status, setStatus] = useState<Status>("idle");
  const [error, setError] = useState<string | null>(null);
  const [recordingSeconds, setRecordingSeconds] = useState(0);

  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<number | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const clearTimer = useCallback(() => {
    if (timerRef.current !== null) {
      window.clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  useEffect(() => {
    return () => {
      clearTimer();
      if (recorderRef.current && recorderRef.current.state !== "inactive") {
        recorderRef.current.stop();
      }
    };
  }, [clearTimer]);

  const start = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      const mimeType = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4"].find(
        (t) => MediaRecorder.isTypeSupported(t),
      );
      const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
      chunksRef.current = [];
      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };
      recorderRef.current = recorder;
      recorder.start();
      setStatus("recording");
      setError(null);
      setRecordingSeconds(0);
      timerRef.current = window.setInterval(() => {
        setRecordingSeconds((s) => s + 1);
      }, 1000);
    } catch (e) {
      setStatus("error");
      setError(e instanceof Error ? e.message : "could not access microphone");
    }
  }, []);

  const stop = useCallback(async (): Promise<Blob | null> => {
    const recorder = recorderRef.current;
    if (!recorder || recorder.state === "inactive") return null;
    clearTimer();

    const mimeType = recorder.mimeType || "audio/webm";
    const blob = await new Promise<Blob | null>((resolve) => {
      recorder.onstop = () => {
        const b = new Blob(chunksRef.current, { type: mimeType });
        chunksRef.current = [];
        resolve(b.size > 0 ? b : null);
      };
      recorder.stop();
    });

    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    setStatus("idle");

    if (!blob) return null;
    try {
      // Decode the compressed WebM/Opus recording to PCM, then package it as a
      // WAV the backend can decode with Symphonia (PCM/WAV features only).
      return await blobToWav(blob);
    } catch (e) {
      setStatus("error");
      setError(e instanceof Error ? e.message : "audio decoding failed");
      return null;
    }
  }, [clearTimer]);

  return { status, error, recordingSeconds, start, stop };
}

/** Decode any supported audio blob to PCM, then encode it as a 16-bit PCM WAV. */
async function blobToWav(blob: Blob): Promise<Blob> {
  const ctx = new AudioContext();
  try {
    const arrayBuffer = await blob.arrayBuffer();
    const audioBuffer = await ctx.decodeAudioData(arrayBuffer);
    return audioBufferToWavBlob(audioBuffer);
  } finally {
    void ctx.close();
  }
}

// --- Streaming recorder (live transcription) ---

export interface StreamRecorderState {
  status: Status;
  error: string | null;
  recordingSeconds: number;
  /** Request the mic and start emitting 16k-mono WAV chunks every ~2 s. */
  start: (onChunk: (wav: Blob) => void) => Promise<void>;
  /** Stop the recorder and emit the final partial chunk. */
  stop: () => Promise<void>;
}

const STREAM_CHUNK_SECS = 2;
const STREAM_RATE = 16000;
const STREAM_CHUNK_LEN = STREAM_CHUNK_SECS * STREAM_RATE;

/**
 * Streaming recorder for live transcription. Unlike `useMediaRecorder` this
 * does NOT use MediaRecorder + `decodeAudioData`: WebAudio cannot reliably
 * decode *partial* WebM/Opus (or partial MP4/AAC in Safari) chunks, which
 * surfaces as "Unable to decode audio data". Instead a ScriptProcessor node
 * captures raw PCM straight off the mic graph, downmixes to mono, resamples to
 * 16 kHz, and re-encodes each ~2 s window directly as a WAV (which the Rust
 * backend — no Opus decoder — can ingest). Emission is synchronous, so `stop()`
 * has nothing to await beyond flushing the tail.
 */
export function useStreamRecorder(): StreamRecorderState {
  const [status, setStatus] = useState<Status>("idle");
  const [error, setError] = useState<string | null>(null);
  const [recordingSeconds, setRecordingSeconds] = useState(0);

  const contextRef = useRef<AudioContext | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const timerRef = useRef<number | null>(null);
  const sendRef = useRef<(wav: Blob) => void>(() => {});
  const chunkBufRef = useRef<Float32Array>(new Float32Array(STREAM_CHUNK_LEN));
  const chunkLenRef = useRef(0);

  const clearTimer = useCallback(() => {
    if (timerRef.current !== null) {
      window.clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  useEffect(() => {
    return () => {
      clearTimer();
      streamRef.current?.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
      void contextRef.current?.close();
      contextRef.current = null;
    };
  }, [clearTimer]);

  const start = useCallback(async (onChunk: (wav: Blob) => void) => {
    const fail = (message: string) => {
      setStatus("error");
      setError(message);
    };

    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (e) {
      fail(e instanceof Error ? e.message : "could not access microphone");
      return;
    }

    try {
      const Ctor: typeof AudioContext =
        window.AudioContext ||
        (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      const ctx = new Ctor();
      await ctx.resume();
      sendRef.current = onChunk;
      chunkBufRef.current = new Float32Array(STREAM_CHUNK_LEN);
      chunkLenRef.current = 0;

      const source = ctx.createMediaStreamSource(stream);
      const processor = ctx.createScriptProcessor(4096, 2, 1);
      processor.onaudioprocess = (e) => {
        const input = e.inputBuffer;
        const inRate = ctx.sampleRate;
        const block = new Float32Array(input.length);
        const channels = Math.min(input.numberOfChannels, 2);
        for (let c = 0; c < channels; c++) {
          const data = input.getChannelData(c);
          for (let i = 0; i < input.length; i++) block[i] += data[i];
        }
        for (let i = 0; i < input.length; i++) block[i] /= channels;
        const mono16 = resampleTo16k(block, inRate);
        accumulateChunk(mono16, ctx, chunkBufRef, chunkLenRef, sendRef.current);
      };
      // Feed the processor output to a muted gain so no mic loopback is heard.
      const mute = ctx.createGain();
      mute.gain.value = 0;
      source.connect(processor);
      processor.connect(mute);
      mute.connect(ctx.destination);

      contextRef.current = ctx;
      sourceRef.current = source;
      processorRef.current = processor;
      streamRef.current = stream;

      setStatus("recording");
      setError(null);
      setRecordingSeconds(0);
      timerRef.current = window.setInterval(() => {
        setRecordingSeconds((s) => s + 1);
      }, 1000);
    } catch (e) {
      stream.getTracks().forEach((t) => t.stop());
      fail(e instanceof Error ? e.message : "audio capture failed");
    }
  }, []);

  const stop = useCallback(async (): Promise<void> => {
    if (!contextRef.current) return;
    clearTimer();
    // Emit whatever is left in the current chunk window.
    const ctx = contextRef.current;
    accumulateChunk(new Float32Array(0), ctx, chunkBufRef, chunkLenRef, sendRef.current, true);
    sourceRef.current?.disconnect();
    processorRef.current?.disconnect();
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    await contextRef.current.close().catch(() => {});
    contextRef.current = null;
    setStatus("idle");
  }, [clearTimer]);

  return { status, error, recordingSeconds, start, stop };
}

/** Downsample a mono block to 16 kHz with linear interpolation. */
function resampleTo16k(input: Float32Array, fromRate: number): Float32Array {
  if (fromRate === STREAM_RATE) return input;
  const ratio = STREAM_RATE / fromRate;
  const outLen = Math.max(1, Math.floor(input.length * ratio));
  const out = new Float32Array(outLen);
  for (let i = 0; i < outLen; i++) {
    const src = (i + 0.5) / ratio - 0.5;
    const i0 = Math.max(0, Math.floor(src));
    const i1 = Math.min(input.length - 1, i0 + 1);
    const frac = Math.min(1, Math.max(0, src - i0));
    out[i] = input[i0] * (1 - frac) + input[i1] * frac;
  }
  return out;
}

/** Fill the current 2 s window; emit a WAV when it fills, or on final flush. */
function accumulateChunk(
  mono16: Float32Array,
  ctx: AudioContext,
  chunkBufRef: { current: Float32Array },
  chunkLenRef: { current: number },
  onChunk: (wav: Blob) => void,
  flush = false,
): void {
  const buf = chunkBufRef.current;
  let len = chunkLenRef.current;
  let offset = 0;
  while (offset < mono16.length) {
    const take = Math.min(STREAM_CHUNK_LEN - len, mono16.length - offset);
    buf.set(mono16.subarray(offset, offset + take), len);
    len += take;
    offset += take;
    if (len === STREAM_CHUNK_LEN) {
      onChunk(wavFromMono16k(new Float32Array(buf), ctx));
      buf.fill(0);
      len = 0;
    }
  }
  if (flush && len > 0) {
    const tail = new Float32Array(len);
    tail.set(buf.subarray(0, len));
    onChunk(wavFromMono16k(tail, ctx));
    len = 0;
  }
  chunkLenRef.current = len;
}

/** Encode raw 16k mono samples directly as a WAV, bypassing decodeAudioData. */
function wavFromMono16k(samples: Float32Array<ArrayBuffer>, ctx: AudioContext): Blob {
  const buffer = ctx.createBuffer(1, samples.length, STREAM_RATE);
  buffer.copyToChannel(samples, 0, 0);
  return audioBufferToWavBlob(buffer);
}

/** Encode an AudioBuffer as a WAV (16-bit PCM), preserving channels/rate. */
function audioBufferToWavBlob(buffer: AudioBuffer): Blob {
  const numChannels = buffer.numberOfChannels;
  const sampleRate = buffer.sampleRate;
  const numFrames = buffer.length;

  const bytesPerSample = 2;
  const blockAlign = numChannels * bytesPerSample;
  const dataSize = numFrames * blockAlign;
  const byteRate = sampleRate * blockAlign;

  const wav = new ArrayBuffer(44 + dataSize);
  const view = new DataView(wav);

  const writeStr = (offset: number, s: string) => {
    for (let i = 0; i < s.length; i++) view.setUint8(offset + i, s.charCodeAt(i));
  };

  writeStr(0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeStr(8, "WAVE");
  writeStr(12, "fmt ");
  view.setUint32(16, 16, true); // fmt chunk size
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, 16, true); // bits per sample
  writeStr(36, "data");
  view.setUint32(40, dataSize, true);

  // Interleave channels into 16-bit signed PCM.
  const channels: Float32Array[] = [];
  for (let c = 0; c < numChannels; c++) channels.push(buffer.getChannelData(c));

  let offset = 44;
  for (let i = 0; i < numFrames; i++) {
    for (let c = 0; c < numChannels; c++) {
      const sample = Math.max(-1, Math.min(1, channels[c][i]));
      const pcm = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
      view.setInt16(offset, pcm, true);
      offset += 2;
    }
  }

  return new Blob([wav], { type: "audio/wav" });
}
