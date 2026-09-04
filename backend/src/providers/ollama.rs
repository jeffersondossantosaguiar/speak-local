use crate::config::Config;
use crate::error::AppError;
use crate::providers::{Analysis, AnalysisProvider, AnalysisResult, ErrorItem};
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
    url: String,
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
   - "explanation": a short plain-English reason, written to help them learn

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
            url: format!("{}/v1/chat/completions", cfg.ollama_url.trim_end_matches('/')),
            model: cfg.llm_model.clone(),
            temperature: cfg.llm_temperature,
            client: reqwest::blocking::Client::new(),
        }
    }

    fn build_request(&self, transcript: &str) -> ChatRequest<'_> {
        let user_content = format!(
            "Here is the transcript of what I spoke to practice English:\n\n\"\"\"\n{transcript}\n\"\"\"\n\nAnalyze it as instructed and return the JSON object."
        );
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
                    content: user_content,
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
            match self
                .client
                .post(&self.url)
                .json(&req)
                .send()
            {
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
}

impl AnalysisProvider for OllamaProvider {
    fn analyze(&self, transcript: &str) -> AnalysisResult {
        self.call_sync(transcript)
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

    let mut errors: Vec<ErrorItem> = json
        .errors
        .into_iter()
        .map(|e| ErrorItem {
            text: e.text.trim().to_string(),
            suggestion: e.suggestion.trim().to_string(),
            category: if e.category.is_empty() {
                "other".to_string()
            } else {
                e.category
            },
            criticality: e.criticality,
            context: e.context.trim().to_string(),
            explanation: e.explanation.trim().to_string(),
        })
        .collect();

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
}
