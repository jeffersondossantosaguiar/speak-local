use crate::config::Config;
use crate::error::AppError;
use crate::providers::{AudioSamples, Transcript, TranscriptionProvider, TranscriptionResult};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

/// Implementation of `TranscriptionProvider` backed by whisper.cpp via
/// whisper-rs. The model context is loaded once at startup and shared;
/// per-call work runs on the blocking pool (see the job runner).
pub struct WhisperProvider {
    ctx: WhisperContext,
    n_threads: usize,
    initial_prompt: String,
}

impl WhisperProvider {
    pub fn new(cfg: &Config) -> Result<Self, AppError> {
        let path = cfg
            .whisper_model
            .to_str()
            .ok_or_else(|| AppError::Config("whisper model path not UTF-8".into()))?;

        let mut params = WhisperContextParameters::default();
        params.use_gpu(cfg.whisper_use_gpu);

        let ctx = WhisperContext::new_with_params(path, params)
            .map_err(|e| AppError::Config(format!("failed to load whisper model '{path}': {e}")))?;

        let n_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
            .min(8);

        Ok(Self {
            ctx,
            n_threads,
            initial_prompt: cfg.whisper_initial_prompt.clone(),
        })
    }
}

impl TranscriptionProvider for WhisperProvider {
    fn transcribe(&self, audio: &AudioSamples) -> TranscriptionResult {
        let mut state = self
            .ctx
            .create_state()
            .map_err(|e| AppError::Transcription(format!("create state: {e}")))?;

        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_language(Some("en"));
        params.set_n_threads(self.n_threads as i32);
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        if !self.initial_prompt.is_empty() {
            params.set_initial_prompt(&self.initial_prompt);
        }

        state
            .full(params, &audio.samples)
            .map_err(|e| AppError::Transcription(format!("inference failed: {e}")))?;

        let n_segments = state.full_n_segments();
        let mut parts = Vec::new();
        for i in 0..n_segments {
            if let Some(seg) = state.get_segment(i) {
                let text = seg.to_string().trim().to_string();
                if !text.is_empty() {
                    parts.push(text);
                }
            }
        }

        let text = parts.join(" ").trim().to_string();
        Ok(Transcript { text })
    }
}
