pub mod ollama;
pub mod whisper;

use crate::error::AppError;
use serde::Serialize;

/// Raw decoded mono PCM samples at 16 kHz (the format Whisper consumes).
pub struct AudioSamples {
    /// Floating point samples in [-1.0, 1.0].
    pub samples: Vec<f32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Transcript {
    pub text: String,
}

pub type TranscriptionResult = Result<Transcript, AppError>;
pub type AnalysisResult = Result<Analysis, AppError>;
pub type SamplesResult = Result<AudioSamples, AppError>;

/// A provider of speech-to-text, behind a small trait so a cloud
/// implementation (e.g. OpenAI Whisper API) can be swapped in later.
pub trait TranscriptionProvider: Send + Sync {
    fn transcribe(&self, samples: &AudioSamples) -> TranscriptionResult;
}

/// One identified language error, most critical first.
#[derive(Debug, Clone, Serialize)]
pub struct ErrorItem {
    /// The original (wrong) text as spoken.
    pub text: String,
    pub suggestion: String,
    /// "grammar", "vocabulary", "pronunciation", "awkward", etc.
    pub category: String,
    /// Higher = more critical. Errors are returned ordered by this descending.
    pub criticality: u32,
    /// The surrounding sentence for context.
    pub context: String,
    /// Why it is wrong, for study value.
    pub explanation: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct Analysis {
    /// CEFR level label (A1..C2), a rough estimate.
    pub cefr_label: String,
    /// Short justification for the level.
    pub cefr_justification: String,
    /// Prioritized error list (most critical first).
    pub errors: Vec<ErrorItem>,
}

/// A provider of grammar/vocab error analysis + CEFR estimate. Small trait so
/// a cloud LLM (Claude API, OpenAI, etc.) can replace the local Ollama impl.
pub trait AnalysisProvider: Send + Sync {
    fn analyze(&self, transcript: &str) -> AnalysisResult;
}
