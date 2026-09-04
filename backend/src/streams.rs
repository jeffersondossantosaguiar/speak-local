use crate::audio::decode_audio;
use crate::config::Config;
use crate::jobs::{JobResult, JobStatus, WhisperGate};
use crate::providers::{AnalysisProvider, AudioSamples, TranscriptionProvider};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};
use uuid::Uuid;

/// What a poll of a streaming session currently looks like.
#[derive(Debug, Clone)]
pub enum StreamView {
    /// Still recording: expose the refining partial draft.
    Active { partial_text: String, audio_seconds: usize },
    /// Finalized: a full job status (processing while the model runs, then
    /// completed/failed).
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
    final_running: bool,
    final_status: Option<JobStatus>,
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
    analysis: Arc<dyn AnalysisProvider>,
    whisper_gate: WhisperGate,
    retention: Duration,
    max_samples: usize,
    partial_interval_samples: usize,
    rms_floor: f32,
}

impl StreamStore {
    pub fn new(
        transcription: Arc<dyn TranscriptionProvider>,
        analysis: Arc<dyn AnalysisProvider>,
        whisper_gate: WhisperGate,
        cfg: &Config,
    ) -> Self {
        Self {
            streams: Arc::new(Mutex::new(HashMap::new())),
            transcription,
            analysis,
            whisper_gate,
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
                    final_running: false,
                    final_status: None,
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

    /// Stop accepting audio and run the full Whisper + LLM pipeline. Idempotent:
    /// calling twice returns early once the final status is stored.
    pub fn finalize(&self, id: Uuid) -> Result<(), String> {
        {
            let streams = self.streams.lock().unwrap();
            let entry = streams.get(&id).ok_or("stream not found")?;
            let mut d = entry.data.lock().unwrap();
            if d.finalized {
                return Ok(());
            }
            if d.final_status.is_some() {
                return Ok(());
            }
            if d.samples.is_empty() {
                return Err("stream has no audio".into());
            }
            if d.samples.len() > self.max_samples {
                d.finalized = true;
                d.final_status = Some(JobStatus::Failed {
                    error: format!("recording exceeds {} seconds", self.max_samples / 16_000),
                });
                return Ok(());
            }
            d.finalized = true;
            d.final_running = true;
        }
        let this = self.clone();
        tokio::spawn(async move {
            if let Err(e) = this.run_final(id).await {
                tracing::warn!(%id, %e, "final pipeline failed");
            }
        });
        Ok(())
    }

    async fn run_final(&self, id: Uuid) -> Result<(), String> {
        let outcome = self.final_status(id).await;
        let mut streams = self.streams.lock().unwrap();
        if let Some(entry) = streams.get_mut(&id) {
            let mut d = entry.data.lock().unwrap();
            d.final_running = false;
            match &outcome {
                Ok(status) => d.final_status = Some(status.clone()),
                Err(e) => {
                    d.final_status = Some(JobStatus::Failed {
                        error: e.clone(),
                    })
                }
            }
        }
        outcome.map(|_| ())
    }

    async fn final_status(&self, id: Uuid) -> Result<JobStatus, String> {
        // Transcribe under the whisper gate (like every other model call).
        let transcript = {
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
                .map_err(|e| e.to_string())?
        };

        // The LLM runs without the whisper gate (see jobs::run_pipeline).
        let analysis = {
            let text = transcript.text.clone();
            let analyzer = self.analysis.clone();
            tokio::task::spawn_blocking(move || analyzer.analyze(&text))
                .await
                .map_err(|e| format!("analysis task panicked: {e}"))?
                .map_err(|e| e.to_string())?
        };

        Ok(JobStatus::Completed {
            result: JobResult { transcript, analysis },
        })
    }

    pub fn get(&self, id: Uuid) -> Option<StreamView> {
        let streams = self.streams.lock().unwrap();
        let entry = streams.get(&id)?;
        let d = entry.data.lock().unwrap();
        if let Some(status) = d.final_status.clone() {
            return Some(StreamView::Finished(status));
        }
        if d.finalized {
            // Final pipeline still in flight.
            return Some(StreamView::Finished(JobStatus::Processing));
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