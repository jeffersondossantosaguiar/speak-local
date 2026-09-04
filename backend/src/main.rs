use axum::extract::DefaultBodyLimit;
use axum::routing::{get, post};
use axum::Router;
use speak_local_backend::config::Config;
use std::path::Path;
use speak_local_backend::handlers::{
    create_stream, finalize_stream, get_job, get_stream, health, stream_ws, submit_job, AppState,
};
use speak_local_backend::jobs::{JobStore, WhisperGate};
use speak_local_backend::providers::ollama::OllamaProvider;
use speak_local_backend::providers::whisper::WhisperProvider;
use speak_local_backend::streams::StreamStore;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Prefer the .env next to the crate so `cargo run` works from the workspace
    // root too (dotenv() alone only searches the CWD and its parents). A .env in
    // the caller's working directory still takes precedence when present.
    dotenvy::from_filename(Path::new(env!("CARGO_MANIFEST_DIR")).join(".env")).ok();
    dotenvy::dotenv().ok();
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,speak_local_backend=debug".into()),
        )
        .init();

    let cfg = Config::from_env()?;
    tracing::info!(
        whisper_model = ?cfg.whisper_model,
        whisper_gpu = cfg.whisper_use_gpu,
        llm_model = %cfg.llm_model,
        ollama_url = %cfg.ollama_url,
        "configuration loaded"
    );

    // Load the (blocking) Whisper model *before* entering the Tokio runtime.
    // whisper.cpp spins up its own native/OpenMP threads; constructing it
    // outside the async runtime avoids a "Cannot drop a runtime in a blocking
    // context" panic at shutdown.
    let transcription: Arc<dyn speak_local_backend::providers::TranscriptionProvider> =
        Arc::new(WhisperProvider::new(&cfg)?);
    let analysis: Arc<dyn speak_local_backend::providers::AnalysisProvider> =
        Arc::new(OllamaProvider::new(&cfg));

    let runtime = tokio::runtime::Runtime::new()?;
    runtime.block_on(run_server(cfg, transcription, analysis))
}

async fn run_server(
    cfg: Config,
    transcription: Arc<dyn speak_local_backend::providers::TranscriptionProvider>,
    analysis: Arc<dyn speak_local_backend::providers::AnalysisProvider>,
) -> Result<(), Box<dyn std::error::Error>> {
    // One whisper gate serializes ALL model access: whole-record jobs and
    // streaming partial/final passes share a single non-thread-safe context.
    let whisper_gate: WhisperGate = Arc::new(Mutex::new(()));
    let job_store = Arc::new(JobStore::with_gate(
        transcription.clone(),
        analysis.clone(),
        &cfg,
        whisper_gate.clone(),
    ));
    let stream_store = Arc::new(StreamStore::new(
        transcription,
        analysis,
        whisper_gate,
        &cfg,
    ));
    let state = AppState {
        jobs: job_store.clone(),
        streams: stream_store.clone(),
    };

    // Periodically sweep finished jobs and stream sessions so the in-memory
    // maps stay bounded.
    {
        let store = job_store.clone();
        let streams = stream_store.clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_secs(60)).await;
                let now = std::time::SystemTime::now();
                store.sweep(now);
                streams.sweep(now);
            }
        });
    }

    let app = Router::new()
        .route("/health", get(health))
        // Uploaded audio is an uncompressed PCM WAV (~11.5 MB/min). The Axum
        // default body limit is 2 MB, which would 413 on any real recording, so
        // raise it well past the longest practical practice clip.
        .route(
            "/jobs",
            post(submit_job).layer(DefaultBodyLimit::max(128 * 1024 * 1024)),
        )
        .route("/jobs/{id}", get(get_job))
        .route("/streams", post(create_stream))
        .route("/streams/{id}", get(get_stream).post(finalize_stream))
        .route("/streams/{id}/ws", get(stream_ws))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(&cfg.bind_addr).await?;
    tracing::info!("listening on http://{}", cfg.bind_addr);
    axum::serve(listener, app).await?;

    Ok(())
}
