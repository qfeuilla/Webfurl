use std::io::{self, Write};

use minillmlib::GeneratorInfo;
use minillmlib::generator::{CompletionParameters, ProviderSettings};
use tokio::io::AsyncBufReadExt;
use tracing::info;

use webfurl_core::{
    cache::CacheStore,
    embeddings::EmbeddingClient,
    pipeline::PipelineConfig,
};

mod agent;
mod browser;

#[tokio::main]
async fn main() {
    dotenvy::dotenv().ok();
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,webfurl_core=debug,webfurl_agent=debug".into()),
        )
        .init();

    let openrouter_key =
        std::env::var("OPENROUTER_API_KEY").expect("OPENROUTER_API_KEY must be set");
    let compression_model = std::env::var("WEBFURL_COMPRESSION_MODEL")
        .expect("WEBFURL_COMPRESSION_MODEL must be set (e.g. openai/gpt-oss-120b). See .env.example");
    let agent_model = std::env::var("WEBFURL_AGENT_MODEL")
        .expect("WEBFURL_AGENT_MODEL must be set (e.g. anthropic/claude-sonnet-4.6). See .env.example");
    let vision_model = std::env::var("WEBFURL_VISION_MODEL")
        .expect("WEBFURL_VISION_MODEL must be set (e.g. google/gemini-2.5-flash). See .env.example");
    let initial_budget: u32 = std::env::var("WEBFURL_TOKEN_BUDGET")
        .expect("WEBFURL_TOKEN_BUDGET must be set (e.g. 5000). See .env.example")
        .parse()
        .expect("WEBFURL_TOKEN_BUDGET must be a valid number");
    let max_budget: u32 = std::env::var("WEBFURL_MAX_BUDGET")
        .expect("WEBFURL_MAX_BUDGET must be set (e.g. 128000). See .env.example")
        .parse()
        .expect("WEBFURL_MAX_BUDGET must be a valid number");

    let embedding_client = EmbeddingClient::new(openrouter_key.clone());

    let compression_generator = GeneratorInfo::openrouter(&compression_model)
        .with_api_key(&openrouter_key)
        .with_default_params(
            CompletionParameters::new()
                .with_provider(ProviderSettings::new().sort_by_throughput()),
        );

    let agent_generator = GeneratorInfo::openrouter(&agent_model)
        .with_api_key(&openrouter_key);

    let vision_generator = GeneratorInfo::openrouter(&vision_model)
        .with_api_key(&openrouter_key)
        .with_vision();

    let pipeline_config = PipelineConfig {
        generator: compression_generator,
        embedding_client: embedding_client.clone(),
        max_depth: 4,
        min_content_length: 10,
    };

    // Connect to MongoDB for caching
    let mongo_uri = std::env::var("MONGODB_URI")
        .unwrap_or_else(|_| "mongodb://localhost:27017".into());
    let cache = CacheStore::new(&mongo_uri)
        .await
        .expect("Failed to connect to MongoDB. Is mongod running?");
    info!("MongoDB cache connected");

    // Launch visible Chrome browser
    let browser_session = browser::BrowserSession::launch()
        .await
        .expect("Failed to launch Chrome browser. Is Chrome/Chromium installed?");

    let mut ctx = agent::AgentContext::new(
        agent_generator,
        pipeline_config,
        embedding_client,
        cache,
        initial_budget,
        max_budget,
        browser_session,
        vision_generator,
    );

    println!("\n=== WebFurl Agent ===");
    println!("Models: compression={compression_model}  agent={agent_model}  vision={vision_model}");
    println!("Live files: webfurl_context.md (LLM context)  webfurl_stats.md (compression stats)");
    println!("Browser: visible (interact freely — changes auto-sync on next prompt)");
    println!("Commands:");
    println!("  /url <url>              Navigate to a URL");
    println!("  /unfold <node_id>       Unfold a node");
    println!("  /fold <node_id>         Fold a node back");
    println!("  /search <query>         Semantic search + auto-unfold");
    println!("  /click <selector>       Click an element");
    println!("  /fill <selector> <text> Type text into an element");
    println!("  /describe <node_id>     Describe images (vision model)");
    println!("  /screenshot             Full-page screenshot");
    println!("  /tree                   Show full debug view");
    println!("  /browser                Open current page in your browser");
    println!("  /quit                   Exit\n");

    let stdin = tokio::io::BufReader::new(tokio::io::stdin());
    let mut lines = stdin.lines();

    loop {
        print!("> ");
        io::stdout().flush().unwrap();

        let input = match lines.next_line().await {
            Ok(Some(text)) if !text.trim().is_empty() => text.trim().to_string(),
            Ok(None) => {
                if let Err(e) = ctx.close_browser().await {
                    eprintln!("Warning: failed to close browser: {e}");
                }
                return;
            }
            _ => continue,
        };

        if input == "/quit" {
            if let Err(e) = ctx.close_browser().await {
                eprintln!("Warning: failed to close browser: {e}");
            }
            break;
        }

        match ctx.handle_input(&input).await {
            Ok(response) if !response.is_empty() => println!("\n{response}\n"),
            Ok(_) => {}
            Err(e) => eprintln!("\nError: {e}\n"),
        }
    }
}
