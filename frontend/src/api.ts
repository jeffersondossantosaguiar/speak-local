export interface Transcript {
  text: string;
}

export interface ErrorItem {
  text: string;
  suggestion: string;
  category: string;
  criticality: number;
  context: string;
  explanation: string;
}

export interface Analysis {
  cefr_label: string;
  cefr_justification: string;
  errors: ErrorItem[];
}

export interface JobResult {
  transcript: Transcript;
  analysis: Analysis;
}

export type JobStatus =
  | { status: "pending" }
  | { status: "processing" }
  | { status: "transcribed"; transcript: Transcript }
  | { status: "completed"; result: JobResult }
  | { status: "failed"; error: string };

export interface SubmitResponse {
  job_id: string;
}

export interface CreateStreamResponse {
  stream_id: string;
}

/** Poll response for an in-progress (recording) streaming session. */
export interface StreamActive {
  status: "active";
  partial_text: string;
  audio_seconds: number;
}

/** Everything GET /streams/{id} can return while recording or after finalize. */
export type StreamStatus = StreamActive | JobStatus;

/** SSE frame from GET /jobs/{id}/analysis/stream: one error box completes. */
export interface AnalysisErrorEvent {
  type: "error";
  error: ErrorItem;
}

/** SSE frame announcing the final, validated analysis. */
export interface AnalysisDoneEvent {
  type: "done";
  analysis: Analysis;
}

export type AnalysisEvent = AnalysisErrorEvent | AnalysisDoneEvent;
