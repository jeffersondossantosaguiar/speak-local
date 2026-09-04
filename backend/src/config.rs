use std::path::{Path, PathBuf};

/// Seed vocabulary for Whisper's initial-prompt bias. Helps technical terms and
/// proper nouns survive transcription (e.g. "NestJS" heard as "NAS.js") by
/// biasing the decoder toward these tokens. Overridable via WHISPER_INITIAL_PROMPT.
const DEFAULT_WHISPER_INITIAL_PROMPT: &str =
    "Node.js, NestJS, TypeScript, JavaScript, AWS, microservices, caching, resilience, Kubernetes, Docker, PostgreSQL, API.";

#[derive(Debug, Clone)]
pub struct Config {
    pub bind_addr: String,
    /// Path to the ggml Whisper model file (.bin).
    pub whisper_model: PathBuf,
    /// Whether whisper.cpp should use the CUDA GPU backend.
    pub whisper_use_gpu: bool,
    /// Seed vocabulary passed to Whisper as an initial prompt.
    pub whisper_initial_prompt: String,
    /// LLM server base URL (Ollama at localhost:11434 by default).
    pub ollama_url: String,
    /// Model name served by the LLM backend.
    pub llm_model: String,
    /// LLM temperature for structured extraction.
    pub llm_temperature: f32,
    /// How long a streaming session is kept before it is swept (seconds).
    pub stream_retention_secs: u64,
    /// Hard cap on recorded audio per session (seconds).
    pub stream_max_secs: usize,
    /// How much new audio triggers a refining partial transcribe (seconds).
    pub stream_partial_interval_secs: usize,
    /// RMS floor below which a chunk is treated as silence and does not trigger
    /// a partial. Keeps Whisper from hallucinating on quiet pauses.
    pub stream_rms_floor: f32,
}

impl Config {
    pub fn from_env() -> Result<Self, Box<dyn std::error::Error>> {
        let whisper_model = std::env::var("WHISPER_MODEL")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("models/ggml-small.en.bin"));

        // Resolve relative model paths against the crate dir so the backend runs
        // from any working directory (workspace root, backend/, etc.).
        let whisper_model = if whisper_model.is_relative() {
            Path::new(env!("CARGO_MANIFEST_DIR")).join(whisper_model)
        } else {
            whisper_model
        };

        let whisper_use_gpu = std::env::var("WHISPER_USE_GPU")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);

        Ok(Self {
            bind_addr: std::env::var("BIND_ADDR").unwrap_or_else(|_| "127.0.0.1:8787".into()),
            whisper_model,
            whisper_use_gpu,
            whisper_initial_prompt: std::env::var("WHISPER_INITIAL_PROMPT")
                .unwrap_or_else(|_| DEFAULT_WHISPER_INITIAL_PROMPT.to_string()),
            ollama_url: std::env::var("OLLAMA_URL").unwrap_or_else(|_| "http://localhost:11434".into()),
            llm_model: std::env::var("LLM_MODEL").unwrap_or_else(|_| "llama3.1:8b".into()),
            llm_temperature: std::env::var("LLM_TEMPERATURE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.1),
            stream_retention_secs: std::env::var("STREAM_RETENTION_SECS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(600),
            stream_max_secs: std::env::var("STREAM_MAX_SECS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(600),
            stream_partial_interval_secs: std::env::var("STREAM_PARTIAL_INTERVAL_SECS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(2),
            stream_rms_floor: std::env::var("STREAM_RMS_FLOOR")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.002),
        })
    }
}
