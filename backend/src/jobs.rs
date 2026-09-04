use crate::audio::decode_audio;
use crate::config::Config;
use crate::error::AppError;
use crate::providers::{
    Analysis, AnalysisProvider, TranscriptionProvider, Transcript, SamplesResult,
};
use serde::Serialize;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};
use uuid::Uuid;

/// A running or finished analysis job.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "status")]
pub enum JobStatus {
    #[serde(rename = "pending")]
    Pending,
    #[serde(rename = "processing")]
    Processing,
    #[serde(rename = "completed")]
    Completed { result: JobResult },
    #[serde(rename = "failed")]
    Failed { error: String },
}

#[derive(Debug, Clone, Serialize)]
pub struct JobResult {
    pub transcript: Transcript,
    pub analysis: Analysis,
}

struct JobEntry {
    state: Arc<Mutex<JobStatus>>,
    created_at: SystemTime,
}

#[derive(Clone)]
pub struct JobStore {
    jobs: Arc<Mutex<HashMap<Uuid, JobEntry>>>,
    transcription: Arc<dyn TranscriptionProvider>,
    analysis: Arc<dyn AnalysisProvider>,
    /// Keep-alive duration for finished jobs before they are swept.
    retention: Duration,
}

impl JobStore {
    pub fn new(
        transcription: Arc<dyn TranscriptionProvider>,
        analysis: Arc<dyn AnalysisProvider>,
        _cfg: &Config,
    ) -> Self {
        Self {
            jobs: Arc::new(Mutex::new(HashMap::new())),
            transcription,
            analysis,
            retention: Duration::from_secs(10 * 60),
        }
    }

    /// Register a new job and kick off the blocking pipeline in the background.
    pub fn submit(&self, audio_bytes: Vec<u8>) -> Uuid {
        let id = Uuid::new_v4();
        let status = Arc::new(Mutex::new(JobStatus::Pending));
        {
            let mut guard = self.jobs.lock().unwrap();
            guard.insert(
                id,
                JobEntry {
                    state: status.clone(),
                    created_at: SystemTime::now(),
                },
            );
        }

        let transcription = self.transcription.clone();
        let analysis = self.analysis.clone();

        tokio::task::spawn_blocking(move || {
            *status.lock().unwrap() = JobStatus::Processing;

            let outcome = run_pipeline(audio_bytes, transcription, analysis);

            let next = match outcome {
                Ok((transcript, analysis)) => JobStatus::Completed {
                    result: JobResult { transcript, analysis },
                },
                Err(e) => JobStatus::Failed {
                    error: e.to_string(),
                },
            };
            *status.lock().unwrap() = next;
        });

        id
    }

    pub fn get(&self, id: Uuid) -> Option<JobStatus> {
        let guard = self.jobs.lock().unwrap();
        guard.get(&id).map(|e| e.state.lock().unwrap().clone())
    }

    /// Drop finished jobs older than the retention window. Called on a timer.
    pub fn sweep(&self, now: SystemTime) {
        let mut guard = self.jobs.lock().unwrap();
        guard.retain(|_, e| {
            let done = matches!(
                *e.state.lock().unwrap(),
                JobStatus::Completed { .. } | JobStatus::Failed { .. }
            );
            if done {
                now.duration_since(e.created_at)
                    .map(|d| d <= self.retention)
                    .unwrap_or(false)
            } else {
                true
            }
        });
    }
}

/// Decode → transcribe → analyze, all in one blocking unit so it runs off the
/// async runtime on a blocking thread.
fn run_pipeline(
    audio_bytes: Vec<u8>,
    transcription: Arc<dyn TranscriptionProvider>,
    analysis: Arc<dyn AnalysisProvider>,
) -> Result<(Transcript, Analysis), AppError> {
    let samples: SamplesResult = decode_audio(&audio_bytes).map_err(AppError::Audio);
    let samples = samples?;
    let transcript = transcription.transcribe(&samples)?;
    let analysis = analysis.analyze(&transcript.text)?;
    Ok((transcript, analysis))
}
