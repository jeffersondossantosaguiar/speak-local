use speak_local_backend::audio::decode_audio;
use speak_local_backend::config::Config;
use speak_local_backend::handlers::health;
use speak_local_backend::jobs::JobStore;
use speak_local_backend::providers::{
    Analysis, AnalysisProvider, AudioSamples, ErrorItem, TranscriptionProvider, Transcript,
};
use std::sync::Arc;

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
    for _ in 0..num_samples {
        wav.extend_from_slice(&0i16.to_le_bytes());
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
