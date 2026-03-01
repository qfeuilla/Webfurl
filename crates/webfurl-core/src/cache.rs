use mongodb::{Client, Collection, IndexModel};
use mongodb::bson::{doc, to_bson, from_bson, Document as BsonDocument};
use mongodb::options::IndexOptions;
use tracing::{debug, info};

use crate::error::WebfurlError;
use crate::tree::SemanticNode;

const DB_NAME: &str = "webfurl";
const CHUNK_COLLECTION: &str = "chunk_cache";
const IMAGE_DESC_COLLECTION: &str = "image_descriptions";
const IMAGE_QUERY_COLLECTION: &str = "image_queries";

#[derive(Clone)]
pub struct CacheStore {
    client: Client,
}

impl CacheStore {
    pub async fn new(mongo_uri: &str) -> crate::Result<Self> {
        let client = Client::with_uri_str(mongo_uri).await?;
        let store = Self { client };
        store.ensure_indexes().await?;
        Ok(store)
    }

    async fn ensure_indexes(&self) -> crate::Result<()> {
        let db = self.client.database(DB_NAME);

        let chunk_col = db.collection::<BsonDocument>(CHUNK_COLLECTION);
        chunk_col
            .create_index(
                IndexModel::builder()
                    .keys(doc! { "content_hash": 1 })
                    .options(IndexOptions::builder().unique(true).build())
                    .build(),
            )
            .await?;

        let img_desc_col = db.collection::<BsonDocument>(IMAGE_DESC_COLLECTION);
        img_desc_col
            .create_index(
                IndexModel::builder()
                    .keys(doc! { "url_hash": 1 })
                    .options(IndexOptions::builder().unique(true).build())
                    .build(),
            )
            .await?;

        let img_query_col = db.collection::<BsonDocument>(IMAGE_QUERY_COLLECTION);
        img_query_col
            .create_index(
                IndexModel::builder()
                    .keys(doc! { "url_hash": 1 })
                    .build(),
            )
            .await?;

        info!("cache indexes ensured");
        Ok(())
    }

    // ─── Chunk cache (content-hash keyed) ───

    /// Look up a cached chunk compression by content hash.
    /// For leaf chunks: returns the complete node (with interactive children).
    /// For parent chunks: returns the node WITHOUT children (they must be re-attached).
    pub async fn get_chunk(&self, content_hash: &str) -> crate::Result<Option<SemanticNode>> {
        let col = self.chunk_collection();
        let filter = doc! { "content_hash": content_hash };

        let result = col.find_one(filter).await?;
        match result {
            Some(doc) => {
                let node: SemanticNode = from_bson(
                    mongodb::bson::Bson::Document(
                        doc.get_document("node")
                            .map_err(|e| WebfurlError::CacheError(e.to_string()))?
                            .clone()
                    )
                ).map_err(|e| WebfurlError::CacheError(e.to_string()))?;
                debug!(hash = content_hash, id = %node.id, "chunk cache hit");
                Ok(Some(node))
            }
            None => Ok(None),
        }
    }

    /// Store a chunk compression result keyed by content hash.
    /// For parent chunks, strip children before storing (they're always fresh).
    pub async fn put_chunk(&self, content_hash: &str, node: &SemanticNode, is_parent: bool) -> crate::Result<()> {
        let col = self.chunk_collection();

        // For parents: store without children (children are always recomputed)
        let store_node = if is_parent && !node.children.is_empty() {
            let mut stripped = node.clone();
            stripped.children = vec![];
            stripped.subtree_token_count = stripped.token_count;
            stripped
        } else {
            node.clone()
        };

        let node_bson = to_bson(&store_node).map_err(|e| WebfurlError::CacheError(e.to_string()))?;
        let doc = doc! {
            "content_hash": content_hash,
            "node": node_bson,
        };

        col.replace_one(doc! { "content_hash": content_hash }, doc)
            .upsert(true)
            .await?;

        Ok(())
    }

    /// Look up a cached image description by URL hash.
    pub async fn get_image_description(&self, url_hash: &str) -> crate::Result<Option<String>> {
        let col = self.image_desc_collection();
        let filter = doc! { "url_hash": url_hash };
        match col.find_one(filter).await? {
            Some(doc) => {
                let desc = doc.get_str("description").unwrap_or_default().to_string();
                debug!(url_hash, "image description cache hit");
                Ok(Some(desc))
            }
            None => Ok(None),
        }
    }

    /// Store an image description.
    pub async fn put_image_description(&self, url_hash: &str, description: &str) -> crate::Result<()> {
        let col = self.image_desc_collection();
        col.replace_one(
            doc! { "url_hash": url_hash },
            doc! { "url_hash": url_hash, "description": description },
        )
        .upsert(true)
        .await?;
        Ok(())
    }

    /// Look up cached image query answers by URL hash.
    /// Returns Vec<(query_embedding, answer)>.
    pub async fn get_image_queries(&self, url_hash: &str) -> crate::Result<Vec<(Vec<f32>, String)>> {
        let col = self.image_query_collection();
        let filter = doc! { "url_hash": url_hash };
        let mut results = vec![];
        let mut cursor = col.find(filter).await?;
        while cursor.advance().await? {
            let doc = cursor.deserialize_current()
                .map_err(|e| WebfurlError::CacheError(e.to_string()))?;
            let answer = doc.get_str("answer").unwrap_or_default().to_string();
            let embedding: Vec<f32> = doc
                .get_array("query_embedding")
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_f64().map(|f| f as f32))
                        .collect()
                })
                .unwrap_or_default();
            if !embedding.is_empty() {
                results.push((embedding, answer));
            }
        }
        Ok(results)
    }

    /// Store an image query answer.
    pub async fn put_image_query(
        &self,
        url_hash: &str,
        query_embedding: &[f32],
        answer: &str,
    ) -> crate::Result<()> {
        let col = self.image_query_collection();
        let embedding_bson: Vec<mongodb::bson::Bson> = query_embedding
            .iter()
            .map(|&f| mongodb::bson::Bson::Double(f as f64))
            .collect();
        col.insert_one(doc! {
            "url_hash": url_hash,
            "query_embedding": embedding_bson,
            "answer": answer,
        })
        .await?;
        Ok(())
    }

    fn chunk_collection(&self) -> Collection<BsonDocument> {
        self.client
            .database(DB_NAME)
            .collection(CHUNK_COLLECTION)
    }

    fn image_desc_collection(&self) -> Collection<BsonDocument> {
        self.client
            .database(DB_NAME)
            .collection(IMAGE_DESC_COLLECTION)
    }

    fn image_query_collection(&self) -> Collection<BsonDocument> {
        self.client
            .database(DB_NAME)
            .collection(IMAGE_QUERY_COLLECTION)
    }
}
