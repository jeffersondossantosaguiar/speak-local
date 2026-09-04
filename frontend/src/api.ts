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
  | { status: "completed"; result: JobResult }
  | { status: "failed"; error: string };

export interface SubmitResponse {
  job_id: string;
}
