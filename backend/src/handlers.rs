use crate::analysis_extract::ErrorObjectExtractor;
use crate::analysis_hub::{AnalysisHub, HubEvent};
use crate::jobs::{JobStatus, JobStore};
use crate::providers::ollama::error_item_from_json_string;
use crate::streams::{StreamStore, StreamView};
use axum::body::Bytes;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::Response;
use axum::Json;
use serde_json::{json, Value};
use std::convert::Infallible;
use std::sync::Arc;
use uuid::Uuid;

/// Combined shared state for the router.
#[derive(Clone)]
pub struct AppState {
    pub jobs: Arc<JobStore>,
    pub streams: Arc<StreamStore>,
    pub hub: Arc<AnalysisHub>,
}

/// POST /jobs — accept an audio upload and start an analysis job.
/// The body is the raw audio bytes (uncompressed WAV). Returns { "job_id": "..." }.
pub async fn submit_job(State(state): State<AppState>, body: Bytes) -> (StatusCode, Json<Value>) {
    if body.is_empty() {
        return (
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(json!({ "error": "empty audio body" })),
        );
    }
    let id = state.jobs.submit(body.to_vec());
    (
        StatusCode::ACCEPTED,
        Json(json!({ "job_id": id.to_string() })),
    )
}

/// GET /jobs/{id} — poll the status of a job. Returns the serialized JobStatus.
pub async fn get_job(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
) -> (StatusCode, Json<Value>) {
    match state.jobs.get(id) {
        Some(status) => (StatusCode::OK, Json(serde_json::to_value(status).unwrap())),
        None => (StatusCode::NOT_FOUND, Json(json!({ "error": "job not found" }))),
    }
}

/// GET /health — liveness probe.
pub async fn health() -> (StatusCode, Json<Value>) {
    (StatusCode::OK, Json(json!({ "status": "ok" })))
}

/// POST /streams — open a new streaming session. Returns { "stream_id": "..." }.
pub async fn create_stream(
    State(state): State<AppState>,
) -> (StatusCode, Json<Value>) {
    let id = state.streams.create();
    (
        StatusCode::CREATED,
        Json(json!({ "stream_id": id.to_string() })),
    )
}

/// GET /streams/{id}/ws — upgrade to a WebSocket; binary messages are WAV
/// chunks (16k mono), a text message "done" ends the recording side.
pub async fn stream_ws(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
    ws: WebSocketUpgrade,
) -> Response {
    ws.on_upgrade(move |socket| handle_stream_socket(socket, state.streams.clone(), id))
}

async fn handle_stream_socket(mut socket: WebSocket, store: Arc<StreamStore>, id: Uuid) {
    while let Some(msg) = socket.recv().await {
        match msg {
            Ok(Message::Binary(bytes)) => {
                if let Err(e) = store.append(id, &bytes) {
                    tracing::debug!(%id, %e, "append to stream failed");
                    break;
                }
            }
            Ok(Message::Text(text)) if text.as_str() == "done" => break,
            Ok(_) => {}
            Err(_) => break,
        }
    }
    tracing::debug!(%id, "stream ws closed");
}

/// GET /streams/{id} — poll a session. While recording this returns
/// { "status": "active", "partial_text", "audio_seconds" }; once finalized the
/// response is the exact JobStatus shape ({ "status": "processing" | "completed" | "failed" }).
pub async fn get_stream(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
) -> (StatusCode, Json<Value>) {
    match state.streams.get(id) {
        Some(StreamView::Active {
            partial_text,
            audio_seconds,
        }) => (
            StatusCode::OK,
            Json(json!({
                "status": "active",
                "partial_text": partial_text,
                "audio_seconds": audio_seconds,
            })),
        ),
        Some(StreamView::Finished(status)) => {
            (StatusCode::OK, Json(serde_json::to_value(status).unwrap()))
        }
        None => (
            StatusCode::NOT_FOUND,
            Json(json!({ "error": "stream not found" })),
        ),
    }
}

/// POST /streams/{id} — stop recording and run the full pipeline. Returns
/// { "status": "finalized", "job_id": "..." }; poll GET /streams/{id} for the
/// transcript/intermediate state and open GET /jobs/{job_id}/analysis/stream
/// for the streaming LLM errors.
pub async fn finalize_stream(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
) -> (StatusCode, Json<Value>) {
    match state.streams.finalize(id) {
        Ok(job) => (
            StatusCode::OK,
            Json(json!({ "status": "finalized", "job_id": job.to_string() })),
        ),
        Err(e) => (StatusCode::BAD_REQUEST, Json(json!({ "error": e }))),
    }
}

/// GET /jobs/{id}/analysis/stream — SSE that replays the analysis generated so
/// far and then streams the LLM output, emitting one event per completed error
/// object and a final `done` with the full validated `Analysis`.
///
/// Events (`data:` lines): `{ "type": "error", "error": { ... } }` and
/// `{ "type": "done", "analysis": { ... } }`.
pub async fn analysis_stream(
    State(state): State<AppState>,
    Path(job): Path<Uuid>,
) -> Response {
    let Some(status) = state.jobs.get(job) else {
        return (
            StatusCode::NOT_FOUND,
            Json(json!({ "error": "job not found" })),
        )
            .into_response();
    };

    let hub = state.hub.clone();
    let jobs = state.jobs.clone();

    let stream = async_stream::stream! {
        let mut extractor = ErrorObjectExtractor::default();

        macro_rules! emit_errors {
            ($slices:expr) => {
                for s in $slices {
                    if let Some(item) = error_item_from_json_string(s) {
                        let data = serde_json::to_string(&json!({ "type": "error", "error": item }))
                            .unwrap_or_default();
                        yield Ok::<_, Infallible>(Event::default().data(data));
                    }
                }
            };
        }

        match &status {
            JobStatus::Completed { result } => {
                let data = serde_json::to_string(&json!({ "type": "done", "analysis": result.analysis }))
                    .unwrap_or_default();
                yield Ok::<_, Infallible>(Event::default().data(data));
                return;
            }
            JobStatus::Failed { .. } => return,
            _ => {}
        }

        let sub = hub.subscribe(job);

        // Replay anything produced before this connection attached, so a late
        // subscriber still sees every box without waiting for new tokens.
        for delta in sub.past_deltas {
            let mut slices = Vec::new();
            extractor.feed(&delta, &mut |slice| slices.push(slice.to_string()));
            emit_errors!(&slices);
        }

        // If the job settled between subscription and now (e.g. done was
        // broadcast before we attached), serve the settled result directly.
        // A job that failed analysis never broadcasts, so close the stream
        // too rather than hanging; the client falls back to polling.
        match jobs.get(job) {
            Some(JobStatus::Completed { result }) => {
                let data =
                    serde_json::to_string(&json!({ "type": "done", "analysis": result.analysis }))
                        .unwrap_or_default();
                yield Ok::<_, Infallible>(Event::default().data(data));
                return;
            }
            Some(JobStatus::Failed { .. }) => return,
            _ => {}
        }

        let mut rx = sub.rx;
        loop {
            match rx.recv().await {
                Ok(HubEvent::Delta(delta)) => {
                    let mut slices = Vec::new();
                    extractor.feed(&delta, &mut |slice| slices.push(slice.to_string()));
                    emit_errors!(&slices);
                }
                Ok(HubEvent::Done(analysis)) => {
                    let data =
                        serde_json::to_string(&json!({ "type": "done", "analysis": analysis }))
                            .unwrap_or_default();
                    yield Ok::<_, Infallible>(Event::default().data(data));
                    return;
                }
                Err(_) => return,
            }
        }
    };

    Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}