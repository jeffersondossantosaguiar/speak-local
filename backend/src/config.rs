use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct Config {
    pub bind_addr: String,
    /// Path to the ggml Whisper model file (.bin).
    pub whisper_model: PathBuf,
    /// Whisper model name used only for logging/diagnostics.
    pub whisper_use_gpu: bool,
    /// LLM server base URL (Ollama at localhost:11434 by default).
    pub ollama_url: String,
    /// Model name served by the LLM backend.
    pub llm_model: String,
    /// LLM temperature for structured extraction.
    pub llm_temperature: f32,
}

impl Config {
    pub fn from_env() -> Result<Self, Box<dyn std::error::Error>> {
        let whisper_model = std::env::var("WHISPER_MODEL")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("models/ggml-small.bin"));

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
            ollama_url: std::env::var("OLLAMA_URL").unwrap_or_else(|_| "http://localhost:11434".into()),
            llm_model: std::env::var("LLM_MODEL").unwrap_or_else(|_| "llama3".into()),
            llm_temperature: std::env::var("LLM_TEMPERATURE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.1),
        })
    }
}
