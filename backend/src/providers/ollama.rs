use crate::config::Config;
use crate::error::AppError;
use crate::providers::{Analysis, AnalysisProvider, AnalysisResult, ErrorItem, strip_markers};
use serde::{Deserialize, Serialize};

/// Local LLM analysis via Ollama's OpenAI-compatible chat endpoint.
///
/// The model is prompted to return a JSON object matching `AnalysisJson`.
/// The raw string is parsed and validated on the backend — never passed
/// straight through to the frontend. A small repair loop retries extraction
/// when the model wraps JSON in prose/code fences.
///
/// Uses a *blocking* client because analysis runs inside `spawn_blocking`;
/// this keeps the trait interface synchronous while the async runtime stays
/// responsive.
pub struct OllamaProvider {
    base_url: String,
    model: String,
    temperature: f32,
    client: reqwest::blocking::Client,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
}

#[derive(Deserialize)]
struct Choice {
    message: Message,
}

#[derive(Deserialize)]
struct Message {
    content: String,
}

/// One NDJSON line of a native `/api/chat` streaming response.
#[derive(Deserialize)]
struct OllamaStreamChunk {
    message: Option<OllamaStreamMessage>,
    done: bool,
}

#[derive(Deserialize)]
struct OllamaStreamMessage {
    content: Option<String>,
}

#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    temperature: f32,
    stream: bool,
    messages: Vec<ChatMessage<'a>>,
    response_format: ResponseFormat,
}

#[derive(Serialize)]
struct ResponseFormat {
    #[serde(rename = "type")]
    typ: &'static str,
}

#[derive(Serialize)]
struct ChatMessage<'a> {
    role: &'a str,
    content: String,
}

#[derive(Deserialize)]
struct AnalysisJson {
    #[serde(rename = "cefr_level")]
    cefr_level: String,
    #[serde(rename = "cefr_justification")]
    cefr_justification: String,
    errors: Vec<ErrorJson>,
}

#[derive(Deserialize)]
struct ErrorJson {
    text: String,
    suggestion: String,
    #[serde(default)]
    category: String,
    #[serde(default)]
    criticality: u32,
    #[serde(default)]
    context: String,
    #[serde(default)]
    explanation: String,
}

const SYSTEM_PROMPT: &str = r#"You are an exacting but fair English teacher for a non-native speaker practicing spoken English. The input is an automatic transcription of their speech and may contain mis-transcribed technical words.

Your job: report ONLY genuine grammatical and vocabulary errors — the kind a proficient native speaker would actually correct out loud. You are scored on precision: a transcription that is grammatically fine should produce an empty "errors" array, and every non-error you report counts against you. So:
- Never flag style, formality, punctuation, sentence flow, or "more formal" phrasing (e.g. "a lot", "every day", "with X", "instead of", "consisted", "daily" are all fine).
- Never flag product names or technical terms as vocabulary errors. If a term in the transcript looks misspelled or invented, flag it at most once with criticality 1-2 as category "other" and note it may be a transcription artifact.
- Never invent errors or quote words that do not appear in the transcript.

Low-confidence spans: some words are wrapped in guillemets, e.g. «NestJS» or «Louisa Labs». Those come from the transcriber, not the speaker: it was unsure what was actually said (typically product names, people names, or specialized jargon). Treat them as PROBABLE transcription artifacts:
- Never build a grammar error, a full-sentence correction, or a rewrite around a marked span.
- If a marked span forms no coherent, plausible phrase, omit it entirely or, at most, report it once as category "other" with criticality 1-2.
- An unmarked word next to a marked span that is genuinely wrong in itself can still be reported normally.

Real errors to catch — criticality 4-5 when they blur meaning:
- wrong verb tense/form or subject-verb agreement
- missing or wrong articles or prepositions
- missing, duplicated, or misordered words
- a word used with the wrong meaning
- phrasing so awkward it would genuinely trip a native listener

For each error include:
   - "text": the exact incorrect wording as it appears in the transcript
   - "suggestion": a complete, correct, grammatical replacement
   - "category": one of "grammar", "vocabulary", "pronunciation", "awkward", "other"
   - "criticality": 1-5 (1-2 for minor nits, 4-5 for meaning-breaking errors)
   - "context": the sentence the error appears in (quote it roughly as spoken)
   - "explanation": a short plain-English reason, written to help them learn. It must describe the defect that genuinely exists in the original "text" AND be fixed by applying "suggestion". Before returning each item, re-check the explanation against the correction: if the explanation claims a change the correction does not actually make (e.g. the suggestion stays in the same tense or the same word order), adjust it until they agree.

Then estimate a rough CEFR level (A1, A2, B1, B2, C1, or C2) and justify it in one sentence ("cefr_justification").

Respond with ONLY a single JSON object, no prose, no markdown code fences. Schema:
{
  "cefr_level": "B1",
  "cefr_justification": "...",
  "errors": [
    { "text": "...", "suggestion": "...", "category": "grammar", "criticality": 5, "context": "...", "explanation": "..." }
  ]
}"#;

impl OllamaProvider {
    pub fn new(cfg: &Config) -> Self {
        Self {
            base_url: cfg.ollama_url.trim_end_matches('/').to_string(),
            model: cfg.llm_model.clone(),
            temperature: cfg.llm_temperature,
            // reqwest's blocking client defaults to a 30 s total timeout, but a
            // full grammar analysis on llama3.1:8b routinely takes 60–90 s on
            // this hardware, so raise it well past the longest practical run.
            client: reqwest::blocking::Client::builder()
                .timeout(std::time::Duration::from_secs(300))
                .build()
                .expect("blocking reqwest client builds"),
        }
    }

    fn chat_url(&self) -> String {
        format!("{}/v1/chat/completions", self.base_url)
    }

    fn stream_url(&self) -> String {
        format!("{}/api/chat", self.base_url)
    }

    fn user_prompt(transcript: &str) -> String {
        format!(
            "Here is the transcript of what I spoke to practice English:\n\n\"\"\"\n{transcript}\n\"\"\"\n\nAnalyze it as instructed and return the JSON object."
        )
    }

    fn build_request(&self, transcript: &str) -> ChatRequest<'_> {
        ChatRequest {
            model: &self.model,
            temperature: self.temperature,
            stream: false,
            messages: vec![
                ChatMessage {
                    role: "system",
                    content: SYSTEM_PROMPT.to_string(),
                },
                ChatMessage {
                    role: "user",
                    content: Self::user_prompt(transcript),
                },
            ],
            response_format: ResponseFormat {
                typ: "json_object",
            },
        }
    }

    /// Send the request. This is IO bound, so callers run it off the async
    /// runtime via spawn_blocking (blocking client) — see the job runner.
    ///
    /// Ollama closes idle keep-alive connections whenever it loads/unloads a
    /// model, and our pooled `reqwest` client can hold a stale connection that
    /// has since been dropped. Those come back as a transport error ("error
    /// sending request") with no HTTP status, so we retry a few times with a
    /// short warm-up before giving up. Non-transport (HTTP status) errors are
    /// returned immediately.
    fn call_sync(&self, transcript: &str) -> Result<Analysis, AppError> {
        const MAX_ATTEMPTS: u32 = 3;
        const WARMUP_MS: u64 = 1500;

        let mut last_err: Option<AppError> = None;
        for attempt in 0..MAX_ATTEMPTS {
            let req = self.build_request(transcript);
            match self.client.post(self.chat_url()).json(&req).send() {
                Ok(resp) => return self.handle_response(resp),
                Err(e) => {
                    last_err = Some(AppError::Analysis(format!("llm request failed: {e}")));
                    let retryable = e.is_connect() || e.is_request();
                    if attempt + 1 < MAX_ATTEMPTS && retryable {
                        std::thread::sleep(std::time::Duration::from_millis(WARMUP_MS));
                        continue;
                    }
                    break;
                }
            }
        }

        Err(last_err.unwrap_or_else(|| AppError::Analysis("llm request failed".into())))
    }

    fn handle_response(
        &self,
        resp: reqwest::blocking::Response,
    ) -> Result<Analysis, AppError> {
        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().unwrap_or_default();
            return Err(AppError::Analysis(format!(
                "llm returned {status}: {body}"
            )));
        }

        let parsed: ChatResponse = resp
            .json()
            .map_err(|e| AppError::Analysis(format!("llm response not parseable: {e}")))?;

        let content = parsed
            .choices
            .first()
            .map(|c| c.message.content.clone())
            .ok_or_else(|| AppError::Analysis("llm returned no choices".into()))?;

        parse_analysis(&content)
    }

    /// Stream the model output over Ollama's native `/api/chat` endpoint and
    /// report each content chunk via `on_delta`, rebuilding the full JSON at
    /// the end for the authoritative result. Same retry philosophy as
    /// [`Self::call_sync`].
    fn stream_sync(
        &self,
        transcript: &str,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<Analysis, AppError> {
        const MAX_ATTEMPTS: u32 = 3;
        const WARMUP_MS: u64 = 1500;

        let mut last_err: Option<AppError> = None;
        for attempt in 0..MAX_ATTEMPTS {
            let body = serde_json::json!({
                "model": self.model,
                "stream": true,
                "format": "json",
                "options": { "temperature": self.temperature },
                "messages": [
                    { "role": "system", "content": SYSTEM_PROMPT },
                    { "role": "user", "content": Self::user_prompt(transcript) },
                ],
            });
            match self.client.post(self.stream_url()).json(&body).send() {
                Ok(resp) => {
                    if !resp.status().is_success() {
                        let status = resp.status();
                        let body = resp.text().unwrap_or_default();
                        return Err(AppError::Analysis(format!(
                            "llm returned {status}: {body}"
                        )));
                    }
                    return self.read_stream(resp, on_delta);
                }
                Err(e) => {
                    last_err = Some(AppError::Analysis(format!("llm stream request failed: {e}")));
                    let retryable = e.is_connect() || e.is_request();
                    if attempt + 1 < MAX_ATTEMPTS && retryable {
                        std::thread::sleep(std::time::Duration::from_millis(WARMUP_MS));
                        continue;
                    }
                    break;
                }
            }
        }

        Err(last_err.unwrap_or_else(|| AppError::Analysis("llm stream request failed".into())))
    }

    fn read_stream(
        &self,
        resp: reqwest::blocking::Response,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<Analysis, AppError> {
        use std::io::BufRead;

        let mut content = String::new();
        let mut reader = std::io::BufReader::new(resp);
        let mut line = String::new();
        loop {
            line.clear();
            let n = reader.read_line(&mut line).map_err(|e| {
                AppError::Analysis(format!("reading llm stream failed: {e}"))
            })?;
            if n == 0 {
                break;
            }
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            // Older Ollama versions end the stream with a bare "done" line.
            if trimmed == "done" {
                break;
            }
            let chunk: OllamaStreamChunk = match serde_json::from_str(trimmed) {
                Ok(c) => c,
                Err(_) => continue,
            };
            if let Some(text) = chunk.message.as_ref().and_then(|m| m.content.as_deref()) {
                if !text.is_empty() {
                    content.push_str(text);
                    on_delta(text);
                }
            }
            if chunk.done {
                break;
            }
        }
        parse_analysis(&content)
    }
}

impl AnalysisProvider for OllamaProvider {
    fn analyze(&self, transcript: &str) -> AnalysisResult {
        self.call_sync(transcript)
    }

    fn analyze_streaming(
        &self,
        transcript: &str,
        on_delta: &mut dyn FnMut(&str),
    ) -> AnalysisResult {
        self.stream_sync(transcript, on_delta)
    }
}

/// Remove common false positives before returning errors to the frontend.
///
/// Local tutor models persistently re-flag acceptable conversational phrasing
/// as "vocabulary" or "awkward" errors (e.g. "a lot" -> "significantly",
/// "consisted" -> "involved", "instead of" -> "rather than", "every day" ->
/// "daily"). These were measured verbatim in our tuning pass. We suppress them
/// *only* when the model did not classify them as a grammar defect, so genuine
/// tense/form/agreement errors always pass through regardless of wording.
fn filter_style_false_positives(errors: Vec<ErrorItem>) -> Vec<ErrorItem> {
    // Lower-cased: "text" | "suggestion" pairs we never want to teach against.
    const KNOWN_FINE: &[&str] = &[
        "a lot",
        "consisted",
        "consisting of",
        "every day",
        "daily",
        "instead of",
        "with one click",
        "with docker",
        "with pricing",
        "for the core",
        "the core",
        "the process is more agile",
    ];

    errors
        .into_iter()
        .filter(|e| {
            let is_grammar = e.category.eq_ignore_ascii_case("grammar");
            let is_pronunciation = e.category.eq_ignore_ascii_case("pronunciation");
            if is_grammar || is_pronunciation {
                return true;
            }
            let text_l = e.text.to_ascii_lowercase();
            let sugg_l = e.suggestion.to_ascii_lowercase();
            !KNOWN_FINE
                .iter()
                .any(|p| text_l.contains(p) || sugg_l.contains(p))
        })
        .map(|mut e| {
            // "awkward" is by definition a stylistic judgment; cap it so noisy
            // nits can't outrank real errors in the UI.
            if e.category.eq_ignore_ascii_case("awkward") {
                e.criticality = e.criticality.min(2);
            }
            e
        })
        .collect()
}

/// Parse the model's JSON response into a validated `Analysis`, tolerating
/// code fences and a leading explanations line. Returns the structured model.
fn parse_analysis(content: &str) -> Result<Analysis, AppError> {
    let cleaned = strip_fences(content);
    let json: AnalysisJson = serde_json::from_str(&cleaned)
        .map_err(|e| AppError::Analysis(format!("llm output was not valid JSON: {e}")))?;

    let mut errors: Vec<ErrorItem> = json.errors.into_iter().map(map_error_json).collect();

    errors = filter_style_false_positives(errors);

    // Guarantee most-critical-first irrespective of model ordering.
    errors.sort_by_key(|e| std::cmp::Reverse(e.criticality));

    let level = json.cefr_level.trim().to_uppercase();
    if !["A1", "A2", "B1", "B2", "C1", "C2"].contains(&level.as_str()) {
        let raw = &json.cefr_level;
        return Err(AppError::Analysis(format!(
            "invalid CEFR level from model: '{raw}'"
        )));
    }

    Ok(Analysis {
        cefr_label: level,
        cefr_justification: json.cefr_justification.trim().to_string(),
        errors,
    })
}

fn map_error_json(e: ErrorJson) -> ErrorItem {
    ErrorItem {
        // The model may quote marked spans back to us; drop the «…» so no
        // transcription marker leaks into the UI.
        text: strip_markers(&e.text).trim().to_string(),
        suggestion: strip_markers(&e.suggestion).trim().to_string(),
        category: if e.category.is_empty() {
            "other".to_string()
        } else {
            e.category
        },
        criticality: e.criticality,
        context: strip_markers(&e.context).trim().to_string(),
        explanation: strip_markers(&e.explanation).trim().to_string(),
    }
}

/// Parse a single streaming JSON object and normalize it into a frontend-ready
/// `ErrorItem` (applying the same denylist + awkward cap as the final parse so
/// a box shown live never contradicts the settled result). Returns `None` when
/// the slice is not a valid error object (e.g. the enclosing document).
pub(crate) fn error_item_from_json_string(slice: &str) -> Option<ErrorItem> {
    let e: ErrorJson = serde_json::from_str(slice).ok()?;
    let item = map_error_json(e);
    filter_style_false_positives(vec![item]).pop()
}

/// Remove leading/trailing ``` markers and any leading prose line so that the
/// JSON can be extracted even when the model doesn't honor the "JSON only"
/// instruction perfectly.
fn strip_fences(content: &str) -> String {
    let trimmed = content.trim();
    let mut body = trimmed
        .strip_prefix("```json")
        .or_else(|| trimmed.strip_prefix("```"))
        .unwrap_or(trimmed);
    if let Some(rest) = body.strip_suffix("```") {
        body = rest;
    }
    // A model may prepend a sentence like "Here is the JSON:" on its own line.
    if let Some(idx) = body.find('{') {
        body = &body[idx..];
    }
    body.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item(criticality: u32, category: &str, text: &str, suggestion: &str) -> ErrorItem {
        ErrorItem {
            text: text.to_string(),
            suggestion: suggestion.to_string(),
            category: category.to_string(),
            criticality,
            context: String::new(),
            explanation: String::new(),
        }
    }

    #[test]
    fn suppresses_known_style_false_positives() {
        let input = vec![
            item(4, "vocabulary", "the response time improved a lot", "improved significantly"),
            item(4, "vocabulary", "consisted of", "consisted involved"),
            item(3, "vocabulary", "I work with these tools every day", "I work with these tools daily"),
            item(4, "awkward", "the team can focus on the product instead of infrastructure", "rather than infrastructure"),
        ];
        let out = filter_style_false_positives(input);
        assert!(out.is_empty());
    }

    #[test]
    fn keeps_real_grammar_and_pronunciation_errors() {
        let input = vec![
            item(5, "grammar", "I used to spent a day fixing issues manually", "I used to spend a day"),
            item(3, "vocabulary", "I make a photo of the sunset", "I take a photo of the sunset"),
            item(3, "pronunciation", "discloyer", "disclosure"),
        ];
        let out = filter_style_false_positives(input);
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].text, "I used to spent a day fixing issues manually");
    }

    #[test]
    fn caps_awkward_criticality() {
        let input = vec![
            item(4, "awkward", "We migrated the system to microservices", "we migrated the system to a microservice architecture"),
        ];
        let out = filter_style_false_positives(input);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].criticality, 2);
    }

    #[test]
    fn strips_confidence_markers_from_model_output() {
        let e = ErrorJson {
            text: "I use «NAS.js» for it".into(),
            suggestion: "I use NestJS for it".into(),
            category: "other".into(),
            criticality: 2,
            context: "«NAS.js» is the new API".into(),
            explanation: "looks like a transcription artifact".into(),
        };
        let mapped = map_error_json(e);
        assert_eq!(mapped.text, "I use NAS.js for it");
        assert_eq!(mapped.suggestion, "I use NestJS for it");
        assert_eq!(mapped.context, "NAS.js is the new API");
        assert_eq!(mapped.explanation, "looks like a transcription artifact");
    }
}
