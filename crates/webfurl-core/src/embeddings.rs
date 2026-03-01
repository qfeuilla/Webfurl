use reqwest::Client;
use serde::{Deserialize, Serialize};
use tracing::debug;

use crate::error::WebfurlError;

const OPENROUTER_EMBEDDINGS_URL: &str = "https://openrouter.ai/api/v1/embeddings";
const DEFAULT_MODEL: &str = "qwen/qwen3-embedding-8b";

#[derive(Clone)]
pub struct EmbeddingClient {
    client: Client,
    api_key: String,
    model: String,
}

#[derive(Serialize)]
struct EmbeddingRequest {
    model: String,
    input: Vec<String>,
    encoding_format: String,
}

#[derive(Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
}

#[derive(Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
    #[allow(dead_code)]
    index: usize,
}

impl EmbeddingClient {
    pub fn new(api_key: String) -> Self {
        Self {
            client: Client::new(),
            api_key,
            model: DEFAULT_MODEL.to_string(),
        }
    }

    pub fn with_model(mut self, model: &str) -> Self {
        self.model = model.to_string();
        self
    }

    /// Compute embeddings for a batch of texts in a single API call.
    pub async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, WebfurlError> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        debug!(count = texts.len(), model = %self.model, "computing embeddings");

        let request = EmbeddingRequest {
            model: self.model.clone(),
            input: texts.to_vec(),
            encoding_format: "float".to_string(),
        };

        let response = self
            .client
            .post(OPENROUTER_EMBEDDINGS_URL)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(WebfurlError::EmbeddingError(format!(
                "OpenRouter returned {status}: {body}"
            )));
        }

        let result: EmbeddingResponse = response.json().await?;

        let mut embeddings: Vec<(usize, Vec<f32>)> = result
            .data
            .into_iter()
            .map(|d| (d.index, d.embedding))
            .collect();
        embeddings.sort_by_key(|(idx, _)| *idx);

        Ok(embeddings.into_iter().map(|(_, emb)| emb).collect())
    }

    /// Compute embedding for a single text.
    pub async fn embed(&self, text: &str) -> Result<Vec<f32>, WebfurlError> {
        let results = self.embed_batch(&[text.to_string()]).await?;
        results
            .into_iter()
            .next()
            .ok_or_else(|| WebfurlError::EmbeddingError("empty response".to_string()))
    }
}

/// Cosine similarity between two vectors.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || b.is_empty() {
        tracing::warn!(a_len = a.len(), b_len = b.len(), "cosine_similarity: empty vector(s)");
        return 0.0;
    }
    if a.len() != b.len() {
        tracing::warn!(a_len = a.len(), b_len = b.len(), "cosine_similarity: DIMENSION MISMATCH");
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if mag_a == 0.0 || mag_b == 0.0 {
        return 0.0;
    }
    dot / (mag_a * mag_b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_identical_vectors() {
        let v = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_orthogonal_vectors() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-6);
    }
}
