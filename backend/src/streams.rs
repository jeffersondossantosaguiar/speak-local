use crate::audio::decode_audio;
use crate::config::Config;
use crate::jobs::{JobStatus, JobStore, WhisperGate};
use crate::providers::{AudioSamples, TranscriptionProvider};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};
use uuid::Uuid;

/// What a poll of a streaming session currently looks like.
#[derive(Debug, Clone)]
pub enum StreamView {
    /// Still recording: expose the refining partial draft.
    Active { partial_text: String, audio_seconds: usize },
    /// Finalized: the delegated job's status (transcribed while whisper
    /// finished and the LLM streams, then completed/failed).
    Finished(JobStatus),
}

#[derive(Debug)]
struct StreamData {
    samples: Vec<f32>,
    /// How much of the buffer has already been covered by a partial pass.
    partial_covered: usize,
    partial_running: bool,
    partial_text: Option<String>,
    finalized: bool,
    /// The job that owns the final pipeline, created by [`StreamStore::finalize`].
    final_job_id: Option<Uuid>,
    /// Set when finalize is rejected (e.g. the recording is too long).
    final_error: Option<String>,
}

#[derive(Debug)]
struct StreamEntry {
    data: Mutex<StreamData>,
    created_at: SystemTime,
}

#[derive(Clone)]
pub struct StreamStore {
    streams: Arc<Mutex<HashMap<Uuid, StreamEntry>>>,
    transcription: Arc<dyn TranscriptionProvider>,
    whisper_gate: WhisperGate,
    /// The final pipeline (whisper + streaming LLM) runs as a job here so it
    /// shares the same analysis hub and intermediate states as uploads.
    jobs: Arc<JobStore>,
    retention: Duration,
    max_samples: usize,
    partial_interval_samples: usize,
    rms_floor: f32,
}

impl StreamStore {
    pub fn new(
        transcription: Arc<dyn TranscriptionProvider>,
        jobs: Arc<JobStore>,
        whisper_gate: WhisperGate,
        cfg: &Config,
    ) -> Self {
        Self {
            streams: Arc::new(Mutex::new(HashMap::new())),
            transcription,
            whisper_gate,
            jobs,
            retention: Duration::from_secs(cfg.stream_retention_secs),
            max_samples: cfg.stream_max_secs * 16_000,
            partial_interval_samples: cfg.stream_partial_interval_secs * 16_000,
            rms_floor: cfg.stream_rms_floor,
        }
    }

    pub fn create(&self) -> Uuid {
        let id = Uuid::new_v4();
        let mut streams = self.streams.lock().unwrap();
        streams.insert(
            id,
            StreamEntry {
                data: Mutex::new(StreamData {
                    samples: Vec::new(),
                    partial_covered: 0,
                    partial_running: false,
                    partial_text: None,
                    finalized: false,
                    final_job_id: None,
                    final_error: None,
                }),
                created_at: SystemTime::now(),
            },
        );
        id
    }

    /// Append a chunk of audio (WAV bytes, 16k mono) to the session buffer.
    /// Triggers a refining partial transcribe when enough new speech has
    /// accumulated.
    pub fn append(&self, id: Uuid, wav_bytes: &[u8]) -> Result<(), String> {
        let decoded = decode_audio(wav_bytes).map_err(|e| e.to_string())?;
        let chunk_rms = rms(&decoded.samples);

        let (trigger, buffer_len) = {
            let streams = self.streams.lock().unwrap();
            let entry = streams.get(&id).ok_or("stream not found")?;
            let mut d = entry.data.lock().unwrap();
            if d.finalized {
                // Late-arriving chunks after finalize are dropped.
                return Ok(());
            }
            d.samples.extend_from_slice(&decoded.samples);
            let should_run = !d.partial_running
                && d.samples.len() - d.partial_covered >= self.partial_interval_samples
                && chunk_rms >= self.rms_floor;
            (should_run, d.samples.len())
        };

        if trigger {
            let mut streams = self.streams.lock().unwrap();
            if let Some(entry) = streams.get_mut(&id) {
                entry.data.lock().unwrap().partial_running = true;
            }
            drop(streams);
            let this = self.clone();
            tokio::spawn(async move {
                if let Err(e) = this.run_partial(id, buffer_len).await {
                    tracing::warn!(%id, %e, "partial transcribe failed");
                }
            });
        }
        Ok(())
    }

    /// Kick off a refining transcribe of the whole buffer so far. The draft is
    /// the full text each time ("draft que refina"), never an append.
    async fn run_partial(&self, id: Uuid, snapshot_len: usize) -> Result<(), String> {
        let text_result = {
            let samples = {
                let streams = self.streams.lock().unwrap();
                let entry = match streams.get(&id) {
                    Some(e) => e,
                    None => return Err("stream not found".into()),
                };
                let data = entry.data.lock().unwrap();
                let out = data.samples.clone();
                drop(data);
                out
            };
            let _g = self.whisper_gate.lock().await;
            let transcriber = self.transcription.clone();
            tokio::task::spawn_blocking(move || transcriber.transcribe(&AudioSamples { samples }))
                .await
                .map_err(|e| format!("transcribe task panicked: {e}"))?
                .map_err(|e| e.to_string())
                .map(|t| t.text)
        };

        let mut streams = self.streams.lock().unwrap();
        if let Some(entry) = streams.get_mut(&id) {
            let mut d = entry.data.lock().unwrap();
            if !d.finalized {
                if let Ok(text) = &text_result {
                    d.partial_text = Some(text.clone());
                    if snapshot_len > d.partial_covered {
                        d.partial_covered = snapshot_len;
                    }
                }
            }
            d.partial_running = false;
        }
        text_result.map(|_| ())
    }

    /// Stop accepting audio and hand the full buffer to the job store, which
    /// runs the Whisper + streaming LLM pipeline and keys the analysis SSE hub
    /// by the returned job id. Idempotent: a second call returns the same id.
    pub fn finalize(&self, id: Uuid) -> Result<Uuid, String> {
        let mut streams = self.streams.lock().unwrap();
        let entry = streams.get_mut(&id).ok_or("stream not found")?;
        let d = entry.data.get_mut().unwrap();

        if let Some(job) = d.final_job_id {
            return Ok(job);
        }
        if let Some(err) = &d.final_error {
            return Err(err.clone());
        }
        if d.samples.is_empty() {
            return Err("stream has no audio".into());
        }
        if d.samples.len() > self.max_samples {
            let msg = format!("recording exceeds {} seconds", self.max_samples / 16_000);
            d.finalized = true;
            d.final_error = Some(msg.clone());
            return Err(msg);
        }

        let samples = std::mem::take(&mut d.samples);
        d.finalized = true;
        let job = self.jobs.submit_from_samples(AudioSamples { samples });
        d.final_job_id = Some(job);
        Ok(job)
    }

    pub fn get(&self, id: Uuid) -> Option<StreamView> {
        let streams = self.streams.lock().unwrap();
        let entry = streams.get(&id)?;
        let d = entry.data.lock().unwrap();
        if let Some(err) = &d.final_error {
            return Some(StreamView::Finished(JobStatus::Failed {
                error: err.clone(),
            }));
        }
        if d.finalized {
            // The final pipeline runs as a job; report its live status.
            return match d.final_job_id.and_then(|jid| self.jobs.get(jid)) {
                Some(status) => Some(StreamView::Finished(status)),
                None => Some(StreamView::Finished(JobStatus::Processing)),
            };
        }
        Some(StreamView::Active {
            partial_text: d.partial_text.clone().unwrap_or_default(),
            audio_seconds: d.samples.len() / 16_000,
        })
    }

    /// Drop sessions older than the retention window (finished or abandoned).
    pub fn sweep(&self, now: SystemTime) {
        let mut streams = self.streams.lock().unwrap();
        streams.retain(|_, e| {
            now.duration_since(e.created_at)
                .map(|d| d <= self.retention)
                .unwrap_or(false)
        });
    }
}

fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum: f32 = samples.iter().map(|s| s * s).sum();
    (sum / samples.len() as f32).sqrt()
}