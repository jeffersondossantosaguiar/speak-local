import { useState } from "react";
import type { Analysis, JobStatus } from "./api";
import { submitAudio, waitForJob } from "./client";
import { useMediaRecorder } from "./useMediaRecorder";
import "./app.css";

const MAX_UPLOAD_BYTES = 120 * 1024 * 1024;

export default function App() {
  const recorder = useMediaRecorder();
  const [phase, setPhase] = useState<"idle" | "analyzing">("idle");
  const [error, setError] = useState<string | null>(null);
  const [statusText, setStatusText] = useState<string>("");
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [transcript, setTranscript] = useState<string>("");

  const handleStop = async () => {
    const blob = await recorder.stop();
    if (!blob) return;
    if (blob.size > MAX_UPLOAD_BYTES) {
      setError("Recording is too large — keep clips under ~5 minutes.");
      return;
    }
    setPhase("analyzing");
    setError(null);
    setAnalysis(null);
    setTranscript("");
    try {
      const jobId = await submitAudio(blob);
      setStatusText("job submitted, waiting for transcription…");
      await waitForJob(
        jobId,
        (s) => {
          setStatusText(describe(s));
        },
        (final) => {
          if (final.status === "completed") {
            const { transcript, analysis: a } = final.result;
            setTranscript(transcript.text);
            setAnalysis(a);
          } else {
            setError(final.error);
          }
        },
      );
    } catch (e) {
      setError(e instanceof Error ? e.message : "request failed");
    } finally {
      setPhase("idle");
    }
  };

  const canRecord = recorder.status !== "recording" && phase !== "analyzing";

  return (
    <main className="app">
      <h1>Speak Local</h1>
      <p className="subtitle">Record yourself, then get transcript + error analysis + CEFR estimate.</p>

      <section className="recorder">
        {recorder.status === "recording" ? (
          <>
            <span className="rec-dot" aria-hidden />
            <span>Recording… {recorder.recordingSeconds}s</span>
            <button onClick={() => void handleStop()}>Stop & Analyze</button>
          </>
        ) : (
          <button disabled={!canRecord} onClick={() => void recorder.start()}>
            Start Recording
          </button>
        )}
        {recorder.error && <p className="error">{recorder.error}</p>}
      </section>

      {phase === "analyzing" && <p className="status">{statusText}</p>}
      {error && <p className="error">{error}</p>}

      {transcript && (
        <section className="results">
          <h2>Your transcript</h2>
          <p className="transcript">“{transcript}”</p>
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