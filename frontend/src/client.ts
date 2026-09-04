import type {
  Analysis,
  AnalysisDoneEvent,
  AnalysisErrorEvent,
  CreateStreamResponse,
  ErrorItem,
  JobStatus,
  StreamActive,
  StreamStatus,
  SubmitResponse,
} from "./api";

/** Upload a WAV (decoded in the browser) and return the created job id. */
export async function submitAudio(audio: Blob): Promise<string> {
  const resp = await fetch("/jobs", {
    method: "POST",
    body: audio,
  });
  if (!resp.ok) {
    const body = await resp.json().catch(() => ({}));
    if (resp.status === 413) {
      throw new Error("Recording is too large — keep clips under ~5 minutes.");
    }
    throw new Error(body.error ?? `upload failed (${resp.status})`);
  }
  const data: SubmitResponse = await resp.json();
  return data.job_id;
}

/** Poll the status of a job. Returns the raw JobStatus. */
export async function getJob(id: string): Promise<JobStatus> {
  const resp = await fetch(`/jobs/${id}`);
  if (resp.status === 404) {
    return { status: "failed", error: "job not found (expired)" };
  }
  if (!resp.ok) {
    throw new Error(`poll failed (${resp.status})`);
  }
  return (await resp.json()) as JobStatus;
}

const POLL_INTERVAL_MS = 1000;

/** Poll until the job reaches a terminal state (completed or failed). */
export async function waitForJob(
  id: string,
  onUpdate: (status: JobStatus) => void,
  onDone: (status: JobStatus & ({ status: "completed" } | { status: "failed" })) => void,
): Promise<void> {
  // eslint-disable-next-line no-constant-condition
  while (true) {
    const status = await getJob(id);
    onUpdate(status);
    if (status.status === "completed" || status.status === "failed") {
      onDone(status);
      return;
    }
    await new Promise((r) => setTimeout(r, POLL_INTERVAL_MS));
  }
}

// --- Streaming sessions ---

/** Create a streaming session and return its id. */
export async function createStream(): Promise<string> {
  const resp = await fetch("/streams", { method: "POST" });
  if (!resp.ok) throw new Error(`failed to start stream (${resp.status})`);
  const data: CreateStreamResponse = await resp.json();
  return data.stream_id;
}

/** Open the WebSocket a session's audio chunks are sent over. */
export function openStreamSocket(id: string): WebSocket {
  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  return new WebSocket(`${proto}://${window.location.host}/streams/${id}/ws`);
}

/** Send the finalize signal; the full pipeline then runs on the backend.
 * Returns the underlying job id, which keys the analysis SSE stream. */
export async function finalizeStream(id: string): Promise<string> {
  const resp = await fetch(`/streams/${id}`, { method: "POST" });
  if (!resp.ok) {
    const body = await resp.json().catch(() => ({}));
    throw new Error(body.error ?? `finalize failed (${resp.status})`);
  }
  const data: { job_id?: string } = await resp.json();
  if (!data.job_id) throw new Error("finalize did not return a job id");
  return data.job_id;
}

/** Poll a session: active (recording draft) or a full JobStatus once finalized. */
export async function getStream(id: string): Promise<StreamStatus> {
  const resp = await fetch(`/streams/${id}`);
  if (resp.status === 404) {
    return { status: "failed", error: "stream not found (expired)" };
  }
  if (!resp.ok) throw new Error(`poll failed (${resp.status})`);
  return (await resp.json()) as StreamStatus;
}

/** Poll until the stream is finalized and terminal; reports the live draft and
 * the transcript as soon as Whisper finishes (LLM errors stream separately). */
export async function waitForStream(
  id: string,
  onPartial: (draft: string) => void,
  onTranscribed: (text: string) => void,
  onTerminal: (status: JobStatus & ({ status: "completed" } | { status: "failed" })) => void,
): Promise<void> {
  // eslint-disable-next-line no-constant-condition
  while (true) {
    const status = await getStream(id);
    if (status.status === "active") {
      onPartial((status as StreamActive).partial_text);
    } else if (status.status === "transcribed") {
      onTranscribed(status.transcript.text);
    } else if (status.status === "completed" || status.status === "failed") {
      onTerminal(status);
      return;
    }
    await new Promise((r) => setTimeout(r, POLL_INTERVAL_MS));
  }
}

// --- Streaming analysis (SSE) ---

/**
 * Subscribe to a job's analysis SSE stream: renders each completed error box
 * as the LLM produces it, then the final validated analysis. Returns an abort
 * function. On any failure the caller should fall back to the job's completed
 * result (polled normally).
 */
export function streamAnalysis(
  jobId: string,
  onError: (error: ErrorItem) => void,
  onDone: (analysis: Analysis) => void,
): () => void {
  const controller = new AbortController();
  void (async () => {
    try {
      const resp = await fetch(`/jobs/${jobId}/analysis/stream`, {
        signal: controller.signal,
        headers: { Accept: "text/event-stream" },
      });
      if (!resp.ok || !resp.body) return;
      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        let idx: number;
        while ((idx = buffer.indexOf("\n\n")) !== -1) {
          const frame = buffer.slice(0, idx);
          buffer = buffer.slice(idx + 2);
          const dataLine = frame
            .split("\n")
            .map((l) => l.trim())
            .find((l) => l.startsWith("data:"));
          if (!dataLine) continue;
          const payload = dataLine.slice("data:".length).trim();
          if (!payload) continue;
          let msg: unknown;
          try {
            msg = JSON.parse(payload);
          } catch {
            continue;
          }
          const event = msg as { type?: string };
          if (event.type === "error") {
            onError((msg as AnalysisErrorEvent).error);
          } else if (event.type === "done") {
            onDone((msg as AnalysisDoneEvent).analysis);
            controller.abort();
            return;
          }
        }
      }
    } catch {
      // Aborted or network hiccup; the poll loop supplies the final result.
    }
  })();
  return () => controller.abort();
}
