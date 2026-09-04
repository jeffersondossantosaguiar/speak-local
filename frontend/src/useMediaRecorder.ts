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
