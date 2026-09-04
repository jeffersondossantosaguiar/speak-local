import { useEffect, useRef, useState } from "react";
import type { Analysis, ErrorItem, JobStatus } from "./api";
import {
  createStream,
  finalizeStream,
  getStream,
  openStreamSocket,
  streamAnalysis,
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
  const analysisAbortRef = useRef<(() => void) | null>(null);
  const [phase, setPhase] = useState<"idle" | "analyzing">("idle");
  const [error, setError] = useState<string | null>(null);
  const [statusText, setStatusText] = useState<string>("");
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [liveErrors, setLiveErrors] = useState<ErrorItem[]>([]);
  const [draft, setDraft] = useState<string>("");
  const [streamId, setStreamId] = useState<string | null>(null);

  const addLiveError = (e: ErrorItem) =>
    setLiveErrors((prev) =>
      prev.some((x) => x.text === e.text && x.suggestion === e.suggestion)
        ? prev
        : [...prev, e],
    );

  const runAnalysisStream = (jobId: string) => {
    analysisAbortRef.current?.();
    analysisAbortRef.current = streamAnalysis(jobId, addLiveError, (a) => {
      setAnalysis(a);
      setLiveErrors([]);
    });
  };

  const abortAnalysisStream = () => {
    analysisAbortRef.current?.();
    analysisAbortRef.current = null;
  };

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

  const handleStart = async () => {
    setError(null);
    setDraft("");
    setAnalysis(null);
    setLiveErrors([]);

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
      const sid = streamId;
      await streamRec.stop();
      wsRef.current?.send("done");
      wsRef.current?.close();
      wsRef.current = null;
      setStreamId(null);

      let jobId: string;
      try {
        jobId = await finalizeStream(sid);
      } catch (e) {
        setError(e instanceof Error ? e.message : "finalize failed");
        setPhase("idle");
        return;
      }

      setPhase("analyzing");
      setStatusText("transcribing & analyzing…");
      // Error boxes stream in as the LLM produces them.
      runAnalysisStream(jobId);
      await waitForStream(
        sid,
        (partial) => setDraft(partial),
        (text) => {
          setDraft(text);
          setStatusText("transcribed — analyzing errors…");
        },
        (final) => {
          if (final.status === "completed") {
            setDraft(final.result.transcript.text);
            setAnalysis(final.result.analysis);
            setLiveErrors([]);
          } else {
            setError(final.error);
          }
          abortAnalysisStream();
          setPhase("idle");
        },
      );
      return;
    }

    const blob = await recorder.stop();
    if (!blob) {
      setPhase("idle");
      return;
    }
    if (blob.size > MAX_UPLOAD_BYTES) {
      setError("Recording is too large — keep clips under ~5 minutes.");
      return;
    }

    let jobId: string;
    try {
      jobId = await submitAudio(blob);
    } catch (e) {
      setError(e instanceof Error ? e.message : "upload failed");
      setPhase("idle");
      return;
    }

    setPhase("analyzing");
    setStatusText("job submitted, waiting for transcription…");
    runAnalysisStream(jobId);
    await waitForJob(
      jobId,
      (s) => setStatusText(describe(s)),
      (final) => {
        if (final.status === "completed") {
          setDraft(final.result.transcript.text);
          setAnalysis(final.result.analysis);
          setLiveErrors([]);
        } else {
          setError(final.error);
        }
        abortAnalysisStream();
        setPhase("idle");
      },
    );
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

      {(analysis || liveErrors.length > 0) && (
        <section className="results">
          <h2>Results</h2>
          {analysis && (
            <div className="cefr">
              <strong>{analysis.cefr_label}</strong>
              <span>{analysis.cefr_justification}</span>
            </div>
          )}
          <ErrorList errors={analysis ? analysis.errors : liveErrors} settled={!!analysis} />
        </section>
      )}
    </main>
  );
}

function describe(s: JobStatus): string {
  switch (s.status) {
    case "pending":
      return "Queued…";
    case "processing":
      return "Transcribing & analyzing…";
    case "transcribed":
      return "Transcribed — analyzing errors…";
    case "completed":
      return "Done.";
    case "failed":
      return `Failed: ${s.error}`;
  }
}

/** The error boxes. `settled` gates the "no errors" empty state so it is only
 * shown once the final analysis actually reports nothing. */
function ErrorList({ errors, settled }: { errors: ErrorItem[]; settled: boolean }) {
  if (errors.length === 0) {
    return settled ? <p className="no-errors">No errors detected 🎉</p> : null;
  }
  return (
    <ol className="errors">
      {errors.map((e, i) => (
        <li key={`${e.text}-${e.suggestion}-${i}`} className={`error-item cat-${e.category}`}>
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
  );
}