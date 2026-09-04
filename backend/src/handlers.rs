use crate::jobs::JobStore;
use axum::body::Bytes;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::Json;
use serde_json::{json, Value};
use std::sync::Arc;
use uuid::Uuid;

pub type SharedJobStore = Arc<JobStore>;

/// POST /jobs — accept an audio upload and start an analysis job.
/// The body is the raw audio bytes (WebM/Opus). Returns { "job_id": "..." }.
pub async fn submit_job(
    State(store): State<SharedJobStore>,
    body: Bytes,
) -> (StatusCode, Json<Value>) {
    if body.is_empty() {
        return (
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(json!({ "error": "empty audio body" })),
        );
    }
    let id = store.submit(body.to_vec());
    (
        StatusCode::ACCEPTED,
        Json(json!({ "job_id": id.to_string() })),
    )
}

/// GET /jobs/{id} — poll the status of a job. Returns the serialized JobStatus.
pub async fn get_job(
    State(store): State<SharedJobStore>,
    Path(id): Path<Uuid>,
) -> (StatusCode, Json<Value>) {
    match store.get(id) {
        Some(status) => (StatusCode::OK, Json(serde_json::to_value(status).unwrap())),
        None => (StatusCode::NOT_FOUND, Json(json!({ "error": "job not found" }))),
    }
}

/// GET /health — liveness probe.
pub async fn health() -> (StatusCode, Json<Value>) {
    (StatusCode::OK, Json(json!({ "status": "ok" })))
}
