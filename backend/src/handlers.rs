use crate::jobs::JobStore;
use crate::streams::{StreamStore, StreamView};
use axum::body::Bytes;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::Response;
use axum::Json;
use serde_json::{json, Value};
use std::sync::Arc;
use uuid::Uuid;

/// Combined shared state for the router.
#[derive(Clone)]
pub struct AppState {
    pub jobs: Arc<JobStore>,
    pub streams: Arc<StreamStore>,
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
/// { "status": "finalized" }; poll GET /streams/{id} for the result.
pub async fn finalize_stream(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
) -> (StatusCode, Json<Value>) {
    match state.streams.finalize(id) {
        Ok(()) => (StatusCode::OK, Json(json!({ "status": "finalized" }))),
        Err(e) => (
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": e })),
        ),
    }
}