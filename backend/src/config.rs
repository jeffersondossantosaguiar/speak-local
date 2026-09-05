use std::path::{Path, PathBuf};

/// Seed vocabulary for Whisper's initial-prompt bias. Helps technical terms and
/// proper nouns survive transcription (e.g. "NestJS" heard as "NAS.js") by
/// biasing the decoder toward these tokens. Deliberately domain-general, so
/// anyone's voice/brand/stack survives without hardcoding (see below:
/// WHISPER_VOCAB_HINTS is the hook for personal terms). Overridable entirely
/// via WHISPER_INITIAL_PROMPT.
const DEFAULT_WHISPER_INITIAL_PROMPT: &str =
    "APIs, microservices, databases, SQL, NoSQL, cloud, WebSockets, REST, CI/CD, Docker, Kubernetes, deployment, TypeScript, JavaScript, React, Node.js, frontend, backend, testing, architecture.";

#[derive(Debug, Clone)]
pub struct Config {
    pub bind_addr: String,
    /// Path to the ggml Whisper model file (.bin).
    pub whisper_model: PathBuf,
    /// Whether whisper.cpp should use the CUDA GPU backend.
    pub whisper_use_gpu: bool,
    /// Seed vocabulary passed to Whisper as an initial prompt.
    pub whisper_initial_prompt: String,
    /// Optional extra terms (company/product names, niche jargon) appended to
    /// the Whisper seed. See [`Self::effective_whisper_prompt`].
    pub whisper_vocab_hints: String,
    /// Tokens with confidence below this are treated as likely transcription
    /// artifacts in the analysis (`WHISPER_LOW_CONF_THRESHOLD`).
    pub whisper_low_conf_threshold: f32,
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
    /// The Whisper decoder seed: the base `WHISPER_INITIAL_PROMPT` (or the
    /// generic default) plus the caller's `WHISPER_VOCAB_HINTS`, comma-separated.
    /// Personal names/products in the hints bend the decoder toward them
    /// without widening the shipped default.
    pub fn effective_whisper_prompt(&self) -> String {
        if self.whisper_vocab_hints.is_empty() {
            self.whisper_initial_prompt.clone()
        } else {
            format!("{}, {}", self.whisper_initial_prompt, self.whisper_vocab_hints)
        }
    }

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
            whisper_vocab_hints: std::env::var("WHISPER_VOCAB_HINTS").unwrap_or_default(),
            whisper_low_conf_threshold: std::env::var("WHISPER_LOW_CONF_THRESHOLD")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.6),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn effective_prompt_composes_base_and_hints() {
        let cfg = Config {
            bind_addr: "x".into(),
            whisper_model: PathBuf::from("m"),
            whisper_use_gpu: false,
            whisper_initial_prompt: "APIs, NoSQL".into(),
            whisper_vocab_hints: "LuizaLabs, LIGA FACENS".into(),
            whisper_low_conf_threshold: 0.6,
            ollama_url: "http://localhost:11434".into(),
            llm_model: "llama3.1:8b".into(),
            llm_temperature: 0.1,
            stream_retention_secs: 600,
            stream_max_secs: 600,
            stream_partial_interval_secs: 2,
            stream_rms_floor: 0.002,
        };
        assert_eq!(
            cfg.effective_whisper_prompt(),
            "APIs, NoSQL, LuizaLabs, LIGA FACENS"
        );
        let mut no_hints = cfg.clone();
        no_hints.whisper_vocab_hints.clear();
        assert_eq!(no_hints.effective_whisper_prompt(), "APIs, NoSQL");
    }

    #[test]
    fn default_whisper_prompt_has_no_personal_names() {
        let cfg = Config {
            bind_addr: "x".into(),
            whisper_model: PathBuf::from("m"),
            whisper_use_gpu: false,
            whisper_initial_prompt: DEFAULT_WHISPER_INITIAL_PROMPT.into(),
            whisper_vocab_hints: String::new(),
            whisper_low_conf_threshold: 0.6,
            ollama_url: "http://localhost:11434".into(),
            llm_model: "llama3.1:8b".into(),
            llm_temperature: 0.1,
            stream_retention_secs: 600,
            stream_max_secs: 600,
            stream_partial_interval_secs: 2,
            stream_rms_floor: 0.002,
        };
        let p = cfg.effective_whisper_prompt();
        assert!(p.contains("APIs"));
        assert!(p.contains("NoSQL"));
        for personal in ["NestJS", "LuizaLabs", "LinkApi", "BuildOne", "Wellhub"] {
            assert!(!p.contains(personal), "default must not contain {personal}");
        }
    }
}
