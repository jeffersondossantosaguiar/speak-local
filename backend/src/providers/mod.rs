pub mod ollama;
pub mod whisper;

use crate::error::AppError;
use serde::Serialize;

/// Raw decoded mono PCM samples at 16 kHz (the format Whisper consumes).
pub struct AudioSamples {
    /// Floating point samples in [-1.0, 1.0].
    pub samples: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct Transcript {
    pub text: String,
    /// Byte ranges into `text` that the transcriber was unsure about (low
    /// per-token confidence). The analysis prompt wraps these in «…» so the LLM
    /// treats them as likely transcription artifacts rather than genuine
    /// speaker errors. Empty for providers without confidence data.
    #[serde(default)]
    pub low_confidence_spans: Vec<(usize, usize)>,
}

impl Transcript {
    /// The text with low-confidence spans wrapped in «…», suitable for feeding
    /// the LLM. The clean [`Self::text`] is kept for display; markers never
    /// appear in the final transcript.
    pub fn analysis_text(&self) -> String {
        mark_spans(&self.text, &self.low_confidence_spans)
    }
}

/// Left/right guillemets used to flag low-confidence spans to the LLM.
const MARK_OPEN: char = '«';
const MARK_CLOSE: char = '»';

/// Wrap the given byte ranges of `text` in «…». Spans are first snapped to
/// whole whitespace-delimited words (so a BPE token like "NestJ" covers the
/// full word "NestJS") and any span that covers no word (a lone space or
/// punctuation) is dropped, since it can only be stray transcription noise.
fn mark_spans(text: &str, spans: &[(usize, usize)]) -> String {
    if spans.is_empty() || text.is_empty() {
        return text.to_string();
    }

    let mut snapped: Vec<(usize, usize)> = Vec::new();
    for &(raw_s, raw_e) in spans {
        let raw_s = raw_s.min(text.len());
        let raw_e = raw_e.min(text.len());
        if raw_s >= raw_e {
            continue;
        }
        // Trim trailing whitespace inside the span to find its last real char.
        let mut inner_end = raw_e;
        while inner_end > raw_s && text.as_bytes()[inner_end - 1].is_ascii_whitespace() {
            inner_end -= 1;
        }
        if inner_end <= raw_s {
            continue; // span holds only whitespace (e.g. a stray " " token)
        }
        // Snap to whole words so a BPE fragment like "NestJ" covers "NestJS".
        let word_start = text.as_bytes()[..raw_s]
            .iter()
            .rposition(|&b| b == b' ')
            .map(|p| p + 1)
            .unwrap_or(0);
        let word_end = text.as_bytes()[inner_end..]
            .iter()
            .position(|&b| b == b' ')
            .map(|p| inner_end + p)
            .unwrap_or(text.len());
        if word_start >= word_end {
            continue;
        }
        let word = &text[word_start..word_end];
        if !word.chars().any(|c| c.is_alphanumeric()) {
            continue; // pure punctuation like a low-confidence "," token
        }
        match snapped.last_mut() {
            Some(last) if word_start <= last.1 => last.1 = last.1.max(word_end),
            _ => snapped.push((word_start, word_end)),
        }
    }

    let mut out = String::with_capacity(text.len() + snapped.len() * 2);
    let mut next = 0;
    for &(s, e) in &snapped {
        if next > s {
            continue;
        }
        out.push_str(&text[next..s]);
        out.push(MARK_OPEN);
        out.push_str(&text[s..e]);
        out.push(MARK_CLOSE);
        next = e;
    }
    out.push_str(&text[next..]);
    out
}

/// Remove «…» markers from model output so they never leak into the UI.
/// Cheap and safe: the markers are fixed ASCII-ish punctuation that never
/// appears in normal speech.
pub(crate) fn strip_markers(s: &str) -> String {
    s.replace([MARK_OPEN, MARK_CLOSE], "")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn transcript(text: &str, spans: &[(usize, usize)]) -> Transcript {
        Transcript {
            text: text.to_string(),
            low_confidence_spans: spans.to_vec(),
        }
    }

    #[test]
    fn marks_low_confidence_spans() {
        let t = transcript(
            "I use NestJS to build microservices",
            &[(6, 12)], // "NestJS"
        );
        assert_eq!(t.analysis_text(), "I use «NestJS» to build microservices");
    }

    #[test]
    fn marks_multiple_and_later_spans() {
        let t = transcript(
            "We deploy via LinkApi and Kafka",
            &[(14, 21), (26, 31)], // "LinkApi", "Kafka"
        );
        assert_eq!(t.analysis_text(), "We deploy via «LinkApi» and «Kafka»");
    }

    #[test]
    fn merges_overlapping_spans() {
        let t = transcript("abcdef", &[(1, 4), (3, 6)]);
        assert_eq!(t.analysis_text(), "«abcdef»");
    }

    #[test]
    fn snaps_spans_to_whole_words() {
        // BPE can split a word ("NestJ" + "S"); the marked region must cover
        // the full word so the LLM sees «NestJS», not «NestJ»S.
        let t = transcript("I use NestJS here", &[(6, 11)]);
        assert_eq!(t.analysis_text(), "I use «NestJS» here");
    }

    #[test]
    fn drops_stray_punctuation_and_space_spans() {
        // A low-confidence "," or " " token is noise; marking it adds nothing.
        let t = transcript("CI pipeline, so engineers", &[(2, 3), (12, 13)]);
        assert_eq!(t.analysis_text(), "CI pipeline, so engineers");
    }

    #[test]
    fn clamps_out_of_range_spans() {
        let t = transcript("hi there", &[(0, 99)]);
        assert_eq!(t.analysis_text(), "«hi there»");
    }

    #[test]
    fn no_spans_leaves_text_unchanged() {
        let t = transcript("perfectly clean speech", &[]);
        assert_eq!(t.analysis_text(), "perfectly clean speech");
    }

    #[test]
    fn strips_markers_unescapes_text() {
        assert_eq!(strip_markers("I use «NAS.js» for it"), "I use NAS.js for it");
    }

    #[test]
    fn default_transcript_is_empty_and_unmarked() {
        let t = Transcript::default();
        assert_eq!(t.text, "");
        assert!(t.low_confidence_spans.is_empty());
        assert_eq!(t.analysis_text(), "");
    }
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

    /// Same as [`Self::analyze`] but reports chunks of the model output to
    /// `on_delta` as they are generated, so the UI can render error boxes
    /// incrementally. The default implementation ignores the callback and
    /// behaves exactly like [`Self::analyze`].
    fn analyze_streaming(
        &self,
        transcript: &str,
        on_delta: &mut dyn FnMut(&str),
    ) -> AnalysisResult {
        let _ = on_delta;
        self.analyze(transcript)
    }
}
