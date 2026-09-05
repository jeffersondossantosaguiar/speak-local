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
    /// Tokens with `token_probability < threshold` are low-confidence.
    low_conf_threshold: f32,
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
            initial_prompt: cfg.effective_whisper_prompt(),
            low_conf_threshold: cfg.whisper_low_conf_threshold,
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

        // Assemble the standard segment text while recording byte ranges of
        // tokens whisper was unsure about. Special tokens (e.g. "[_BEG_]",
        // "<|endoftext|>") are decoded as text starting with "[" or "<|", so
        // they are skipped rather than joined into the transcript.
        let mut text = String::new();
        let mut spans: Vec<(usize, usize)> = Vec::new();
        let mut in_span = false;
        let mut span_start = 0;
        for i in 0..n_segments {
            let Some(seg) = state.get_segment(i) else {
                continue;
            };
            for t in 0..seg.n_tokens() {
                let Some(tok) = seg.get_token(t) else {
                    continue;
                };
                let Ok(raw) = tok.to_str() else {
                    continue;
                };
                let piece = raw.trim_start();
                if piece.is_empty() || piece.starts_with('[') || piece.contains("<|") {
                    continue;
                }
                let had_leading_space = piece.len() < raw.len();
                let low = tok.token_probability() < self.low_conf_threshold;

                if low {
                    if !in_span {
                        if had_leading_space && !text.ends_with(' ') {
                            text.push(' ');
                        }
                        span_start = text.len();
                        in_span = true;
                    } else if had_leading_space && !text.ends_with(' ') {
                        text.push(' ');
                    }
                } else {
                    if in_span {
                        spans.push((span_start, text.len()));
                        in_span = false;
                    }
                    if had_leading_space && !text.ends_with(' ') {
                        text.push(' ');
                    }
                }
                text.push_str(piece);
            }
        }
        if in_span {
            spans.push((span_start, text.len()));
        }
        let text = text.trim().to_string();
        Ok(Transcript {
            text,
            low_confidence_spans: spans,
        })
    }
}
