import { useEffect, useRef, useState } from "react";
import type { Analysis, JobStatus } from "./api";
import {
  createStream,
  finalizeStream,
  getStream,
  openStreamSocket,
  submitAudio,
  waitForJob,
  waitForStream,
} from "./client";
import { useMediaRecorder, useStreamRecorder } from "./useMediaRecorder";
import "./app.css";

const MAX_UPLOAD_BYTES = 120 * 1024 * 1024;

export default function App() {
  const recorder = useMediaRecorder();
  const streamRec = useStreamRecorder();
  const wsRef = useRef<WebSocket | null>(null);
  const [phase, setPhase] = useState<"idle" | "analyzing">("idle");
  const [error, setError] = useState<string | null>(null);
  const [statusText, setStatusText] = useState<string>("");
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [draft, setDraft] = useState<string>("");
  const [streamId, setStreamId] = useState<string | null>(null);

  // While a session is open, poll for the refining live draft (before the
  // user stops the recording, the final waitForStream poll has not started).
  useEffect(() => {
    if (!streamId) return;
    const tick = async () => {
      try {
        const st = await getStream(streamId);
        if (st.status === "active") setDraft(st.partial_text);
      } catch {
        // Transient poll failures are fine; recording continues.
      }
    };
    void tick();
    const t = window.setInterval(() => void tick(), 1000);
    return () => window.clearInterval(t);
  }, [streamId]);

  const runFinalAnalysis = async (from: () => Promise<void>, finalize: () => Promise<void>) => {
    setPhase("analyzing");
    setError(null);
    setAnalysis(null);
    try {
      await from();
      await finalize();
      setStatusText("transcribing & analyzing…");
      return true;
    } catch (e) {
      setError(e instanceof Error ? e.message : "request failed");
      setPhase("idle");
      return false;
    }
  };

  const handleStart = async () => {
    setError(null);
    setDraft("");
    setAnalysis(null);

    let ws: WebSocket | null = null;
    let sid: string | null = null;
    try {
      sid = await createStream();
      ws = openStreamSocket(sid);
      await new Promise<void>((resolve, reject) => {
        ws!.onopen = () => resolve();
        ws!.onerror = () => reject(new Error("could not open the stream"));
      });
    } catch {
      // Streaming unavailable (backend down, WS proxying missing): fall back to
      // the classic stop-and-upload flow.
      await recorder.start();
      return;
    }
    wsRef.current = ws;
    setStreamId(sid);
    await streamRec.start((wav) => {
      const s = wsRef.current;
      if (s && s.readyState === WebSocket.OPEN) s.send(wav);
    });
    // If the mic permission failes, streamRec.error is shown and the orphan
    // session is swept by the backend retention timer.
  };

  const handleStop = async () => {
    if (streamId) {
      const ok = await runFinalAnalysis(
        async () => {
          await streamRec.stop();
          wsRef.current?.send("done");
          wsRef.current?.close();
          wsRef.current = null;
        },
        () => finalizeStream(streamId),
      );
      if (!ok) return;
      setStreamId(null);
      await waitForStream(
        streamId,
        (partial) => setDraft(partial),
        (final) => {
          if (final.status === "completed") {
            const { transcript, analysis: a } = final.result;
            setDraft(transcript.text);
            setAnalysis(a);
          } else {
            setError(final.error);
          }
        },
      );
      setPhase("idle");
      return;
    }

    const blob = await recorder.stop();
    if (!blob) return;
    if (blob.size > MAX_UPLOAD_BYTES) {
      setError("Recording is too large — keep clips under ~5 minutes.");
      return;
    }
    const ok = await runFinalAnalysis(
      async () => {
        const jobId = await submitAudio(blob);
        setStatusText("job submitted, waiting for transcription…");
        await waitForJob(
          jobId,
          (s) => setStatusText(describe(s)),
          (final) => {
            if (final.status === "completed") {
              const { transcript, analysis: a } = final.result;
              setDraft(transcript.text);
              setAnalysis(a);
            } else {
              setError(final.error);
            }
          },
        );
      },
      async () => {},
    );
    if (ok) setPhase("idle");
  };

  const recording = streamRec.status === "recording" || recorder.status === "recording";
  const canRecord = !recording && phase !== "analyzing";

  return (
    <main className="app">
      <h1>Speak Local</h1>
      <p className="subtitle">Record yourself, then get transcript + error analysis + CEFR estimate.</p>

      <section className="recorder">
        {recording ? (
          <>
            <span className="rec-dot" aria-hidden />
            <span>
              Recording… {streamRec.recordingSeconds || recorder.recordingSeconds}s
            </span>
            <button onClick={() => void handleStop()}>Stop & Analyze</button>
          </>
        ) : (
          <button disabled={!canRecord} onClick={() => void handleStart()}>
            Start Recording
          </button>
        )}
        {recorder.error && <p className="error">{recorder.error}</p>}
        {streamRec.error && <p className="error">{streamRec.error}</p>}
      </section>

      {phase === "analyzing" && <p className="status">{statusText}</p>}
      {error && <p className="error">{error}</p>}

      {draft && (
        <section className="results">
          <h2>Live transcript</h2>
          <p className="transcript draft">“{draft}”</p>
        </section>
      )}

      {analysis && <AnalysisView analysis={analysis} />}
    </main>
  );
}

function describe(s: JobStatus): string {
  switch (s.status) {
    case "pending":
      return "Queued…";
    case "processing":
      return "Transcribing & analyzing…";
    case "completed":
      return "Done.";
    case "failed":
      return `Failed: ${s.error}`;
  }
}

function AnalysisView({ analysis }: { analysis: Analysis }) {
  return (
    <section className="results">
      <h2>Results</h2>
      <div className="cefr">
        <strong>{analysis.cefr_label}</strong>
        <span>{analysis.cefr_justification}</span>
      </div>
      {analysis.errors.length === 0 ? (
        <p className="no-errors">No errors detected 🎉</p>
      ) : (
        <ol className="errors">
          {analysis.errors.map((e, i) => (
            <li key={i} className={`error-item cat-${e.category}`}>
              <div className="error-head">
                <span className="category">{e.category}</span>
                <span className="criticality">criticality {e.criticality}</span>
              </div>
              <p className="context">“{e.context}”</p>
              <p>
                <s>{e.text}</s> → <strong>{e.suggestion}</strong>
              </p>
              {e.explanation && <p className="explanation">{e.explanation}</p>}
            </li>
          ))}
        </ol>
      )}
    </section>
  );
}