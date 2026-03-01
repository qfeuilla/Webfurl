use std::collections::HashMap;
use tracing::{info, warn};

use minillmlib::{ChatNode, GeneratorInfo, CompletionParameters, NodeCompletionParameters, MessageContent, ImageData};

use crate::cache::CacheStore;
use crate::embeddings::{cosine_similarity, EmbeddingClient};
use crate::pipeline::estimate_tokens;
use crate::tree::ImageRef;

const DESCRIBE_PROMPT: &str = r#"You describe images for a web browsing agent that cannot see the page visually.

A [description] is a concise factual summary of what the image contains. A [description] includes:
- Objects, people, layout, and spatial relationships
- All visible text, reproduced exactly
- Visual style and context (photo, icon, chart, illustration)

Output only the [description]. No preamble. Under 100 words.

If the image is broken, blank, or a tracking pixel, output: "[empty or decorative image]""#;

const QUERY_PROMPT_TEMPLATE: &str = r#"You answer questions about images for a web browsing agent.

Question: {question}

Answer the question based only on what is visible in the image. Be concise and factual.

If the answer is not visible in the image, say exactly: "Not visible in this image.""#;

/// Fuzzy similarity threshold for cache hits on directed queries.
const FUZZY_CACHE_THRESHOLD: f32 = 0.92;

/// A cached answer to a directed question about an image.
#[derive(Debug, Clone)]
struct CachedImageQuery {
    query_embedding: Vec<f32>,
    answer: String,
}

/// Vision module for describing images and answering directed questions about them.
/// Caches results by image URL hash, with fuzzy matching for directed queries.
pub struct VisionClient {
    generator: GeneratorInfo,
    embedding_client: EmbeddingClient,
    cache: CacheStore,
    /// In-memory cache: image_url_hash → general description
    description_cache: HashMap<String, String>,
    /// In-memory cache: image_url_hash → vec of (query_embedding, answer) for directed queries
    query_cache: HashMap<String, Vec<CachedImageQuery>>,
}

impl VisionClient {
    pub fn new(generator: GeneratorInfo, embedding_client: EmbeddingClient, cache: CacheStore) -> Self {
        Self {
            generator,
            embedding_client,
            cache,
            description_cache: HashMap::new(),
            query_cache: HashMap::new(),
        }
    }

    /// Generate a general description of an image.
    /// Returns the description and whether it was a cache hit.
    /// Two-tier cache: in-memory → MongoDB → compute.
    pub async fn describe_image(
        &mut self,
        image: &ImageRef,
    ) -> Result<(String, bool), String> {
        // Tier 1: in-memory cache
        if let Some(cached) = self.description_cache.get(&image.url_hash) {
            info!(url_hash = %image.url_hash, "🟢 MEMORY CACHE HIT: image description");
            return Ok((cached.clone(), true));
        }

        // Tier 2: MongoDB cache
        if let Ok(Some(cached)) = self.cache.get_image_description(&image.url_hash).await {
            info!(url_hash = %image.url_hash, "🟢 MONGODB CACHE HIT: image description");
            self.description_cache.insert(image.url_hash.clone(), cached.clone());
            return Ok((cached, true));
        }

        // Tier 3: compute
        info!(url_hash = %image.url_hash, url = %image.url, "🔴 CACHE MISS: generating image description");
        let description = self.call_vision_model(&image.url, DESCRIBE_PROMPT).await?;

        // Store in both caches
        self.description_cache.insert(image.url_hash.clone(), description.clone());
        self.cache.put_image_description(&image.url_hash, &description).await
            .unwrap_or_else(|e| panic!("Failed to persist image description to MongoDB: {e}"));

        info!(url_hash = %image.url_hash, tokens = description.len() / 4, "image description generated and cached");
        Ok((description, false))
    }

    /// Ask a specific question about an image.
    /// Uses fuzzy embedding matching to check if a similar question was already asked.
    /// Two-tier cache: in-memory → MongoDB → compute.
    pub async fn query_image(
        &mut self,
        image: &ImageRef,
        question: &str,
    ) -> Result<(String, bool), String> {
        let query_embedding = self.embedding_client
            .embed(question)
            .await
            .map_err(|e| format!("Embedding error: {e}"))?;

        // Tier 1: in-memory fuzzy cache
        if let Some(cached_queries) = self.query_cache.get(&image.url_hash) {
            for cached in cached_queries {
                let similarity = cosine_similarity(&query_embedding, &cached.query_embedding);
                if similarity >= FUZZY_CACHE_THRESHOLD {
                    info!(url_hash = %image.url_hash, similarity = %format!("{similarity:.3}"), "🟢 MEMORY CACHE HIT: fuzzy image query");
                    return Ok((cached.answer.clone(), true));
                }
            }
        }

        // Tier 2: MongoDB fuzzy cache (load all queries for this image)
        if !self.query_cache.contains_key(&image.url_hash) {
            if let Ok(mongo_queries) = self.cache.get_image_queries(&image.url_hash).await {
                if !mongo_queries.is_empty() {
                    let cached_vec: Vec<CachedImageQuery> = mongo_queries
                        .into_iter()
                        .map(|(emb, ans)| CachedImageQuery { query_embedding: emb, answer: ans })
                        .collect();
                    // Check for fuzzy match in MongoDB results
                    let mut hit_answer: Option<String> = None;
                    for cached in &cached_vec {
                        let similarity = cosine_similarity(&query_embedding, &cached.query_embedding);
                        if similarity >= FUZZY_CACHE_THRESHOLD {
                            info!(url_hash = %image.url_hash, similarity = %format!("{similarity:.3}"), "🟢 MONGODB CACHE HIT: fuzzy image query");
                            hit_answer = Some(cached.answer.clone());
                            break;
                        }
                    }
                    self.query_cache.insert(image.url_hash.clone(), cached_vec);
                    if let Some(answer) = hit_answer {
                        return Ok((answer, true));
                    }
                }
            }
        }

        // Tier 3: compute
        info!(url_hash = %image.url_hash, url = %image.url, question = %question, "🔴 CACHE MISS: querying vision model");
        let prompt = QUERY_PROMPT_TEMPLATE.replace("{question}", question);
        let answer = self.call_vision_model(&image.url, &prompt).await?;

        // Store in both caches
        self.query_cache
            .entry(image.url_hash.clone())
            .or_default()
            .push(CachedImageQuery {
                query_embedding: query_embedding.clone(),
                answer: answer.clone(),
            });
        self.cache.put_image_query(&image.url_hash, &query_embedding, &answer).await
            .unwrap_or_else(|e| panic!("Failed to persist image query to MongoDB: {e}"));

        info!(url_hash = %image.url_hash, tokens = answer.len() / 4, "image query answered and cached");
        Ok((answer, false))
    }

    /// Call the vision model with an image URL and a text prompt.
    async fn call_vision_model(
        &self,
        image_url: &str,
        prompt: &str,
    ) -> Result<String, String> {
        let root = ChatNode::root(prompt);

        // Build multimodal content: text + image URL
        let image_data = ImageData::from_url(image_url);
        let content = MessageContent::with_images("Please analyze this image.", &[image_data]);
        let user_node = root.add_user(content);

        let params = NodeCompletionParameters::new()
            .with_retry(2)
            .with_params(
                CompletionParameters::new()
                    .with_max_tokens(300)
                    .with_temperature(0.1),
            );

        let result = user_node
            .complete(&self.generator, Some(&params))
            .await
            .map_err(|e| format!("Vision model error: {e}"))?;

        let response = result.text().unwrap_or_default().to_string();

        if response.is_empty() {
            warn!(image_url, "vision model returned empty response");
            return Err("Vision model returned empty response".into());
        }

        Ok(response)
    }

    /// Bulk describe: take a mutable tree and fill in descriptions for all undescribed images.
    /// Only images without a description (other than alt text default) get processed.
    pub async fn describe_all_images(
        &mut self,
        images: &mut [ImageRef],
    ) -> Result<DescribeStats, String> {
        let mut stats = DescribeStats::default();

        for image in images.iter_mut() {
            // Skip images that already have a non-alt description
            // (alt text is set as default during extraction, we upgrade those too)
            if image.description.is_some() && image.description != Some(image.alt.clone()) {
                stats.already_described += 1;
                continue;
            }

            match self.describe_image(image).await {
                Ok((description, cache_hit)) => {
                    let tokens = estimate_tokens(&description);
                    image.description = Some(description);
                    image.description_tokens = tokens;
                    if cache_hit {
                        stats.cache_hits += 1;
                    } else {
                        stats.computed += 1;
                    }
                }
                Err(e) => {
                    warn!(url = %image.url, error = %e, "failed to describe image, keeping alt text");
                    stats.errors += 1;
                }
            }
        }

        Ok(stats)
    }
}

#[derive(Debug, Default)]
pub struct DescribeStats {
    pub already_described: usize,
    pub cache_hits: usize,
    pub computed: usize,
    pub errors: usize,
}

impl std::fmt::Display for DescribeStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} already described, {} cache hits, {} computed, {} errors",
            self.already_described, self.cache_hits, self.computed, self.errors
        )
    }
}
