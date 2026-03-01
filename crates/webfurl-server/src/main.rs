use axum::{
    extract::{Json, State},
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tower_http::cors::CorsLayer;
use tracing::info;

use webfurl_core::{
    cache::CacheStore,
    embeddings::EmbeddingClient,
    pipeline::{self, PipelineConfig},
    serialize::{self, CollapsedPage},
    unfold::{self, UnfoldState},
    tree::SemanticTree,
};

struct AppState {
    cache: CacheStore,
    pipeline_config: PipelineConfig,
}

#[tokio::main]
async fn main() {
    dotenvy::dotenv().ok();
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,webfurl_core=debug".into()),
        )
        .init();

    let openrouter_key =
        std::env::var("OPENROUTER_API_KEY").expect("OPENROUTER_API_KEY must be set");
    let mongo_uri =
        std::env::var("MONGODB_URI").unwrap_or_else(|_| "mongodb://localhost:27017".into());
    let compression_model = std::env::var("WEBFURL_COMPRESSION_MODEL")
        .unwrap_or_else(|_| "anthropic/claude-sonnet-4.6".into());

    let cache = CacheStore::new(&mongo_uri)
        .await
        .expect("failed to connect to MongoDB");

    let embedding_client = EmbeddingClient::new(openrouter_key.clone());

    let generator = minillmlib::GeneratorInfo::openrouter(&compression_model)
        .with_api_key(&openrouter_key);

    let pipeline_config = PipelineConfig {
        generator,
        embedding_client,
        max_depth: 4,
        min_content_length: 10,
    };

    let state = Arc::new(AppState {
        cache,
        pipeline_config,
    });

    let app = Router::new()
        .route("/api/health", get(health))
        .route("/api/compress", post(compress_page))
        .route("/api/unfold", post(unfold_node))
        .route("/api/fold", post(fold_node))
        .route("/api/search", post(semantic_search))
        .route("/api/serialize", post(serialize_context))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let addr = "0.0.0.0:3001";
    info!("unfurl server listening on {addr}");
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn health() -> &'static str {
    "ok"
}

// --- Request/Response types ---

#[derive(Deserialize)]
struct CompressRequest {
    html: String,
    url: String,
}

#[derive(Serialize)]
struct CompressResponse {
    tree: SemanticTree,
    cache_hit: bool,
    compressed_tokens: u32,
    full_tokens: u32,
    compression_ratio: f32,
}

#[derive(Deserialize)]
struct UnfoldRequest {
    tree: SemanticTree,
    state: UnfoldStateDto,
    node_id: String,
}

#[derive(Deserialize)]
struct FoldRequest {
    tree: SemanticTree,
    state: UnfoldStateDto,
    node_id: String,
}

#[derive(Deserialize)]
struct SearchRequest {
    tree: SemanticTree,
    state: UnfoldStateDto,
    query: String,
    max_unfolds: Option<usize>,
}

#[derive(Deserialize)]
struct SerializeRequest {
    tree: SemanticTree,
    state: UnfoldStateDto,
    #[serde(default)]
    collapsed_pages: Vec<CollapsedPageDto>,
}

#[derive(Serialize, Deserialize, Clone)]
struct UnfoldStateDto {
    unfolded: Vec<String>,
    token_usage: u32,
    initial_budget: u32,
    max_budget: u32,
}

#[derive(Serialize, Deserialize, Clone)]
struct CollapsedPageDto {
    url: String,
    domain: String,
    summary: String,
    original_tokens: u32,
}

#[derive(Serialize)]
struct UnfoldResponse {
    state: UnfoldStateDto,
    added_tokens: u32,
}

#[derive(Serialize)]
struct FoldResponse {
    state: UnfoldStateDto,
    reclaimed_tokens: u32,
}

#[derive(Serialize)]
struct SearchResponse {
    state: UnfoldStateDto,
    unfolded_nodes: Vec<String>,
}

#[derive(Serialize)]
struct SerializeResponse {
    context: String,
    token_estimate: u32,
}

// --- Handlers ---

async fn compress_page(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CompressRequest>,
) -> Json<CompressResponse> {
    let (tree, _run_stats) = pipeline::html_to_semantic_tree_cached(
        &req.html,
        &req.url,
        &state.pipeline_config,
        &state.cache,
    )
    .await
    .expect("pipeline failed");

    let ratio = tree.compression_ratio();
    Json(CompressResponse {
        compressed_tokens: tree.compressed_token_count,
        full_tokens: tree.full_token_count,
        compression_ratio: ratio,
        tree,
        cache_hit: false,
    })
}

async fn unfold_node(
    State(_state): State<Arc<AppState>>,
    Json(req): Json<UnfoldRequest>,
) -> Json<UnfoldResponse> {
    let mut uf_state = dto_to_unfold_state(&req.state);
    let added = unfold::unfold_node(&req.tree, &mut uf_state, &req.node_id).unwrap_or(0);

    Json(UnfoldResponse {
        state: unfold_state_to_dto(&uf_state),
        added_tokens: added,
    })
}

async fn fold_node(
    State(_state): State<Arc<AppState>>,
    Json(req): Json<FoldRequest>,
) -> Json<FoldResponse> {
    let mut uf_state = dto_to_unfold_state(&req.state);
    let reclaimed = unfold::fold_node(&req.tree, &mut uf_state, &req.node_id).unwrap_or(0);

    Json(FoldResponse {
        state: unfold_state_to_dto(&uf_state),
        reclaimed_tokens: reclaimed,
    })
}

async fn semantic_search(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SearchRequest>,
) -> Json<SearchResponse> {
    let query_embedding = state
        .pipeline_config
        .embedding_client
        .embed(&req.query)
        .await
        .unwrap_or_default();

    let mut uf_state = dto_to_unfold_state(&req.state);
    let unfolded_nodes = unfold::semantic_unfold(
        &req.tree,
        &mut uf_state,
        &query_embedding,
        req.max_unfolds.unwrap_or(5),
    );

    Json(SearchResponse {
        state: unfold_state_to_dto(&uf_state),
        unfolded_nodes,
    })
}

async fn serialize_context(
    State(_state): State<Arc<AppState>>,
    Json(req): Json<SerializeRequest>,
) -> Json<SerializeResponse> {
    let uf_state = dto_to_unfold_state(&req.state);
    let collapsed: Vec<CollapsedPage> = req
        .collapsed_pages
        .into_iter()
        .map(|p| CollapsedPage {
            url: p.url,
            domain: p.domain,
            summary: p.summary,
            original_tokens: p.original_tokens,
        })
        .collect();

    let context = serialize::serialize_tree(&req.tree, &uf_state, &collapsed);
    let token_estimate = (context.len() as u32) / 4;

    Json(SerializeResponse {
        context,
        token_estimate,
    })
}

// --- DTO conversions ---

fn dto_to_unfold_state(dto: &UnfoldStateDto) -> UnfoldState {
    let mut state = UnfoldState::new(dto.initial_budget, dto.max_budget);
    state.unfolded = dto.unfolded.clone();
    state.token_usage = dto.token_usage;
    state
}

fn unfold_state_to_dto(state: &UnfoldState) -> UnfoldStateDto {
    UnfoldStateDto {
        unfolded: state.unfolded.clone(),
        token_usage: state.token_usage,
        initial_budget: state.initial_budget,
        max_budget: state.max_budget,
    }
}
