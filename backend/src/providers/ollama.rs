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

const SYSTEM_PROMPT: &str = r#"You are a strict English tutor. The user is a non-native English speaker practicing spoken English. Given a transcript of their speech, do two things:

1. List the grammar and vocabulary errors in the transcript, most critical first. For each error include:
   - "text": the original incorrect wording
   - "suggestion": the corrected wording
   - "category": one of "grammar", "vocabulary", "pronunciation", "awkward", "other"
   - "criticality": an integer, higher means more critical
   - "context": the sentence the error appears in (quote it roughly as spoken)
   - "explanation": a short plain-English reason why it is wrong, written to help them learn

If there are no errors, return an empty "errors" array.

2. Estimate a rough CEFR level (A1, A2, B1, B2, C1, or C2) for this speaker, and give a one-sentence justification ("cefr_justification").

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
