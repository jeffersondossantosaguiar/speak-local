use crate::analysis_hub::AnalysisHub;
use crate::audio::decode_audio;
use crate::config::Config;
use crate::error::AppError;
use crate::providers::{
    Analysis, AnalysisProvider, AudioSamples, TranscriptionProvider, Transcript,
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
    /// Whisper has finished; the transcript is available even though the LLM
    /// analysis is still streaming. Lets the UI show the text right away.
    #[serde(rename = "transcribed")]
    Transcribed { transcript: Transcript },
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

/// Serializes access to the Whisper model. whisper.cpp contexts are not
/// thread-safe, so every transcribe call (whole-record jobs *and* streaming
/// partial/final passes) must go through this before touching the model.
pub type WhisperGate = Arc<tokio::sync::Mutex<()>>;

#[derive(Clone)]
pub struct JobStore {
    jobs: Arc<Mutex<HashMap<Uuid, JobEntry>>>,
    transcription: Arc<dyn TranscriptionProvider>,
    analysis: Arc<dyn AnalysisProvider>,
    /// Shared lock that serializes whisper model access across all callers.
    whisper_gate: WhisperGate,
    /// Broadcasts the streaming LLM analysis to SSE subscribers per job.
    hub: Arc<AnalysisHub>,
    /// Keep-alive duration for finished jobs before they are swept.
    retention: Duration,
}

impl JobStore {
    pub fn new(
        transcription: Arc<dyn TranscriptionProvider>,
        analysis: Arc<dyn AnalysisProvider>,
        _cfg: &Config,
    ) -> Self {
        Self::with_gate(
            transcription,
            analysis,
            _cfg,
            Arc::new(tokio::sync::Mutex::new(())),
        )
    }

    /// Same as [`Self::new`], but shares a caller-supplied whisper gate so
    /// streaming sessions and whole-record jobs serialize on the same model.
    pub fn with_gate(
        transcription: Arc<dyn TranscriptionProvider>,
        analysis: Arc<dyn AnalysisProvider>,
        _cfg: &Config,
        whisper_gate: WhisperGate,
    ) -> Self {
        Self::with_hub_and_gate(
            transcription,
            analysis,
            _cfg,
            whisper_gate,
            Arc::new(AnalysisHub::new()),
        )
    }

    /// Full control: shared whisper gate + externally-visible analysis hub.
    pub fn with_hub_and_gate(
        transcription: Arc<dyn TranscriptionProvider>,
        analysis: Arc<dyn AnalysisProvider>,
        _cfg: &Config,
        whisper_gate: WhisperGate,
        hub: Arc<AnalysisHub>,
    ) -> Self {
        Self {
            jobs: Arc::new(Mutex::new(HashMap::new())),
            transcription,
            analysis,
            whisper_gate,
            hub,
            retention: Duration::from_secs(10 * 60),
        }
    }

    /// Register a new job from raw audio bytes and kick off the pipeline in the
    /// background. Decoding happens on the blocking thread, so an undecodable
    /// upload becomes a `Failed` job rather than a synchronous error.
    pub fn submit(&self, audio_bytes: Vec<u8>) -> Uuid {
        let id = Uuid::new_v4();
        let status = self.new_pending(id);

        let transcription = self.transcription.clone();
        let analysis = self.analysis.clone();
        let gate = self.whisper_gate.clone();
        let hub = self.hub.clone();

        tokio::spawn(async move {
            let decoded = tokio::task::spawn_blocking(move || decode_audio(&audio_bytes))
                .await
                .map_err(|e| AppError::Internal(format!("decode task panicked: {e}")));
            match decoded {
                Ok(Ok(samples)) => {
                    run_job(id, status, samples, transcription, analysis, gate, hub).await
                }
                Ok(Err(e)) => *status.lock().unwrap() = JobStatus::Failed {
                    error: e.to_string(),
                },
                Err(e) => *status.lock().unwrap() = JobStatus::Failed {
                    error: e.to_string(),
                },
            }
        });

        id
    }

    /// Register a new job from already-decoded samples (streaming finalize
    /// path). Same pipeline, same whisper gate, same analysis hub.
    pub fn submit_from_samples(&self, samples: AudioSamples) -> Uuid {
        let id = Uuid::new_v4();
        let status = self.new_pending(id);

        let transcription = self.transcription.clone();
        let analysis = self.analysis.clone();
        let gate = self.whisper_gate.clone();
        let hub = self.hub.clone();

        tokio::spawn(async move {
            run_job(id, status, samples, transcription, analysis, gate, hub).await
        });

        id
    }

    fn new_pending(&self, id: Uuid) -> Arc<Mutex<JobStatus>> {
        let status = Arc::new(Mutex::new(JobStatus::Pending));
        let mut guard = self.jobs.lock().unwrap();
        guard.insert(
            id,
            JobEntry {
                state: status.clone(),
                created_at: SystemTime::now(),
            },
        );
        status
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

/// Transcribe → publish the transcript → stream the LLM analysis. The whisper
/// gate is held only across the model call; the LLM analysis runs without it so
/// streaming partials of other sessions are not blocked while one job's
/// language model completes. The `Transcribed` intermediate state lets the UI
/// show the transcript as soon as Whisper finishes, while error boxes arrive
/// over the analysis SSE stream.
#[allow(clippy::too_many_arguments)]
async fn run_job(
    id: Uuid,
    state: Arc<Mutex<JobStatus>>,
    samples: AudioSamples,
    transcription: Arc<dyn TranscriptionProvider>,
    analysis: Arc<dyn AnalysisProvider>,
    gate: WhisperGate,
    hub: Arc<AnalysisHub>,
) {
    let set = |s: JobStatus| *state.lock().unwrap() = s;
    set(JobStatus::Processing);

    let transcript = {
        let _g = gate.lock().await;
        tokio::task::spawn_blocking(move || transcription.transcribe(&samples))
            .await
            .map_err(|e| AppError::Internal(format!("transcribe task panicked: {e}")))
            .and_then(|r| r)
    };
    let transcript = match transcript {
        Ok(t) => t,
        Err(e) => {
            set(JobStatus::Failed {
                error: e.to_string(),
            });
            return;
        }
    };
    set(JobStatus::Transcribed {
        transcript: transcript.clone(),
    });

    let text = transcript.analysis_text();
    let hub2 = hub.clone();
    let analysis = tokio::task::spawn_blocking(move || {
        analysis.analyze_streaming(&text, &mut |delta: &str| {
            hub2.push_delta(id, delta);
        })
    })
    .await
    .map_err(|e| AppError::Internal(format!("analysis task panicked: {e}")))
    .and_then(|r| r);

    match analysis {
        Ok(a) => {
            hub.push_done(id, a.clone());
            set(JobStatus::Completed {
                result: JobResult { transcript, analysis: a },
            });
        }
        Err(e) => set(JobStatus::Failed {
            error: e.to_string(),
        }),
    }
}