use speak_local_backend::audio::decode_audio;
use speak_local_backend::config::Config;
use speak_local_backend::handlers::health;
use speak_local_backend::jobs::JobStore;
use speak_local_backend::providers::{
    Analysis, AnalysisProvider, AudioSamples, ErrorItem, TranscriptionProvider, Transcript,
};
use speak_local_backend::streams::{StreamStore, StreamView};
use std::sync::Arc;
use tokio::sync::Mutex as AsyncMutex;

struct StubTranscriber;

impl TranscriptionProvider for StubTranscriber {
    fn transcribe(&self, _samples: &AudioSamples) -> speak_local_backend::providers::TranscriptionResult {
        Ok(Transcript {
            text: "hello world".into(),
        })
    }
}

struct StubAnalyzer;

impl AnalysisProvider for StubAnalyzer {
    fn analyze(&self, _transcript: &str) -> speak_local_backend::providers::AnalysisResult {
        Ok(Analysis {
            cefr_label: "B1".into(),
            cefr_justification: "stub".into(),
            errors: vec![ErrorItem {
                text: "he go".into(),
                suggestion: "he goes".into(),
                category: "grammar".into(),
                criticality: 5,
                context: "he go to school".into(),
                explanation: "third-person singular".into(),
            }],
        })
    }
}

/// Build a minimal valid mono 16-bit PCM WAV with `seconds` of silence.
fn make_wav(seconds: u32) -> Vec<u8> {
    make_pcm_wav(seconds, |_| 0i16)
}

/// Same, but a 440 Hz tone so it clears an RMS silence floor.
fn make_noisy_wav(seconds: u32) -> Vec<u8> {
    make_pcm_wav(seconds, |i| {
        ((i as f32 * 2.0 * std::f32::consts::PI * 440.0 / 16000.0).sin() * 9000.0) as i16
    })
}

fn make_pcm_wav(seconds: u32, sample: impl Fn(u32) -> i16) -> Vec<u8> {
    let sample_rate = 16000u32;
    let num_samples = sample_rate * seconds;
    let data_len = num_samples * 2;
    let byte_rate = sample_rate * 2;
    let mut wav = Vec::new();
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&(36 + data_len).to_le_bytes());
    wav.extend_from_slice(b"WAVE");
    wav.extend_from_slice(b"fmt ");
    wav.extend_from_slice(&16u32.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes()); // PCM
    wav.extend_from_slice(&1u16.to_le_bytes()); // mono
    wav.extend_from_slice(&sample_rate.to_le_bytes());
    wav.extend_from_slice(&byte_rate.to_le_bytes());
    wav.extend_from_slice(&2u16.to_le_bytes()); // block align
    wav.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_len.to_le_bytes());
    for i in 0..num_samples {
        wav.extend_from_slice(&sample(i).to_le_bytes());
    }
    wav
}

#[test]
fn decodes_wave_audio_to_samples() {
    let wav = make_wav(1);
    let result = decode_audio(&wav).expect("wav should decode");
    assert!(!result.samples.is_empty());
    // Mono 16k: ~1 second of audio.
    assert!(result.samples.len() > 10000);
    assert!(result.samples.len() < 20000);
}

#[tokio::test]
async fn health_returns_ok() {
    let (status, body) = health().await;
    assert_eq!(status, axum::http::StatusCode::OK);
    assert_eq!(body.get("status").and_then(|v| v.as_str()), Some("ok"));
}

#[tokio::test]
async fn full_pipeline_reaches_completed() {
    let cfg = Config::from_env().unwrap();
    let store = JobStore::new(
        Arc::new(StubTranscriber),
        Arc::new(StubAnalyzer),
        &cfg,
    );

    let id = store.submit(make_wav(1));

    // Poll until the background job completes (stub providers are instant).
    let mut final_status = None;
    for _ in 0..50 {
        let status = store.get(id).unwrap();
        if !matches!(status, speak_local_backend::jobs::JobStatus::Processing | speak_local_backend::jobs::JobStatus::Pending) {
            final_status = Some(status);
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }

    let status = final_status.expect("job should reach a terminal state");
    match status {
        speak_local_backend::jobs::JobStatus::Completed { result } => {
            assert_eq!(result.transcript.text, "hello world");
            assert_eq!(result.analysis.cefr_label, "B1");
            assert_eq!(result.analysis.errors.len(), 1);
            assert_eq!(result.analysis.errors[0].suggestion, "he goes");
        }
        other => panic!("expected completed, got {other:?}"),
    }
}

#[tokio::test]
async fn missing_job_returns_not_found() {
    let cfg = Config::from_env().unwrap();
    let store = JobStore::new(
        Arc::new(StubTranscriber),
        Arc::new(StubAnalyzer),
        &cfg,
    );
    let id = uuid::Uuid::new_v4();
    assert!(store.get(id).is_none());
}

fn stream_store(cfg: &Config) -> StreamStore {
    StreamStore::new(
        Arc::new(StubTranscriber),
        Arc::new(StubAnalyzer),
        Arc::new(AsyncMutex::new(())),
        cfg,
    )
}

#[tokio::test]
async fn stream_reaches_completed_after_finalize() {
    let cfg = Config::from_env().unwrap();
    let store = stream_store(&cfg);

    let id = store.create();
    store.append(id, &make_wav(1)).unwrap();
    store.append(id, &make_wav(1)).unwrap();
    if let StreamView::Active { audio_seconds, .. } = store.get(id).unwrap() {
        assert_eq!(audio_seconds, 2);
    } else {
        panic!("stream should be active before finalize");
    }

    store.finalize(id).unwrap();

    let mut final_status = None;
    for _ in 0..50 {
        match store.get(id).unwrap() {
            StreamView::Finished(status) => {
                if !matches!(
                    status,
                    speak_local_backend::jobs::JobStatus::Processing
                ) {
                    final_status = Some(status);
                    break;
                }
            }
            StreamView::Active { .. } => {}
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }

    match final_status.expect("stream should reach a terminal state") {
        speak_local_backend::jobs::JobStatus::Completed { result } => {
            assert_eq!(result.transcript.text, "hello world");
            assert_eq!(result.analysis.cefr_label, "B1");
            assert_eq!(result.analysis.errors[0].suggestion, "he goes");
        }
        other => panic!("expected completed, got {other:?}"),
    }

    // Finalize is idempotent.
    store.finalize(id).unwrap();
}

#[tokio::test]
async fn stream_refines_partial_draft() {
    let mut cfg = Config::from_env().unwrap();
    cfg.stream_partial_interval_secs = 1;
    cfg.stream_rms_floor = 0.0;
    let store = stream_store(&cfg);

    let id = store.create();
    // Non-silence wav so the chunk clears the (zeroed) RMS floor.
    store.append(id, &make_noisy_wav(1)).unwrap();

    let mut draft = None;
    for _ in 0..50 {
        if let StreamView::Active { partial_text, .. } = store.get(id).unwrap() {
            if !partial_text.is_empty() {
                draft = Some(partial_text);
                break;
            }
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }

    assert_eq!(draft.as_deref(), Some("hello world"), "partial draft should refine");
}

#[tokio::test]
async fn stream_append_to_unknown_id_errors() {
    let cfg = Config::from_env().unwrap();
    let store = stream_store(&cfg);
    assert!(store.append(uuid::Uuid::new_v4(), &make_wav(1)).is_err());
}

#[tokio::test]
async fn stream_sweep_keeps_fresh_sessions() {
    let cfg = Config::from_env().unwrap();
    let store = stream_store(&cfg);
    let id = store.create();
    store.append(id, &make_wav(1)).unwrap();
    let now = std::time::SystemTime::now();
    store.sweep(now);
    assert!(store.get(id).is_some(), "fresh stream survives sweep");
    // An entry created far in the past is dropped.
    store.sweep(now + std::time::Duration::from_secs(60 * 60));
    assert!(store.get(id).is_none(), "expired stream is swept");
}

struct FailingTranscriber;

impl TranscriptionProvider for FailingTranscriber {
    fn transcribe(
        &self,
        _samples: &AudioSamples,
    ) -> speak_local_backend::providers::TranscriptionResult {
        Err(speak_local_backend::error::AppError::Transcription(
            "boom".into(),
        ))
    }
}

#[tokio::test]
async fn stream_failure_reaches_failed_not_stuck_processing() {
    let cfg = Config::from_env().unwrap();
    let store = StreamStore::new(
        Arc::new(FailingTranscriber),
        Arc::new(StubAnalyzer),
        Arc::new(AsyncMutex::new(())),
        &cfg,
    );

    let id = store.create();
    store.append(id, &make_wav(1)).unwrap();
    store.finalize(id).unwrap();

    let mut final_status = None;
    for _ in 0..50 {
        match store.get(id).unwrap() {
            StreamView::Finished(status) => {
                if !matches!(status, speak_local_backend::jobs::JobStatus::Processing) {
                    final_status = Some(status);
                    break;
                }
            }
            StreamView::Active { .. } => {}
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }

    match final_status.expect("should reach a terminal state") {
        speak_local_backend::jobs::JobStatus::Failed { error } => {
            assert_eq!(error, "transcription failed: boom");
        }
        other => panic!("expected failed, got {other:?}"),
    }
}
