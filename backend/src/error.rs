use thiserror::Error;

#[derive(Debug, Error)]
pub enum AppError {
    #[error("config error: {0}")]
    Config(String),
    #[error("audio decoding failed: {0}")]
    Audio(String),
    #[error("transcription failed: {0}")]
    Transcription(String),
    #[error("analysis failed: {0}")]
    Analysis(String),
    #[error("unprocessable request: {0}")]
    Unprocessable(String),
    #[error("internal error: {0}")]
    Internal(String),
}

impl AppError {
    pub fn status_code(&self) -> u16 {
        match self {
            AppError::Config(_) | AppError::Internal(_) => 500,
            AppError::Audio(_) | AppError::Transcription(_) | AppError::Analysis(_) => 502,
            AppError::Unprocessable(_) => 422,
        }
    }
}
