use thiserror::Error;

#[derive(Error, Debug)]
pub enum WebfurlError {
    #[error("HTML parsing failed: {0}")]
    ParseError(String),

    #[error("LLM request failed: {0}")]
    LlmError(String),

    #[error("Embedding request failed: {0}")]
    EmbeddingError(String),

    #[error("Cache error: {0}")]
    CacheError(String),

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("Node not found: {0}")]
    NodeNotFound(String),

    #[error("Budget exceeded: requested {requested} tokens, budget {budget}")]
    BudgetExceeded { requested: u32, budget: u32 },

    #[error("HTTP error: {0}")]
    HttpError(#[from] reqwest::Error),

    #[error("MongoDB error: {0}")]
    MongoError(#[from] mongodb::error::Error),

    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),
}
