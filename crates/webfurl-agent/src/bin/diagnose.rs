//! Diagnostic tool: launches Chrome, navigates to a URL, renders it,
//! then runs clean_html + chunk_dom + full LLM compression pipeline.
//!
//! Outputs to Webfurl/diagnose/:
//!   1_raw.html          — raw rendered HTML from Chrome
//!   2_cleaned.html      — after clean_html (noise stripped, attrs cleaned)
//!   3_chunks.txt        — chunk tree structure (pre-LLM)
//!   4_tree.txt          — full semantic tree, fully unfolded (post-LLM)
//!   5_stats.txt         — compression stats
//!
//! Usage: cargo run --release --bin diagnose -- "https://www.airbnb.com/s/..."

use chromiumoxide::browser::{Browser, BrowserConfig};
use dom_query::Document;
use futures::StreamExt;
use std::io::Write;
use webfurl_core::pipeline;
use webfurl_core::tree::SemanticNode;

fn out_dir() -> std::path::PathBuf {
    // Look for Cargo.toml upward to find workspace root
    let mut dir = std::env::current_dir().expect("cwd");
    loop {
        if dir.join("Cargo.toml").exists() && dir.join("crates").exists() {
            return dir.join("diagnose");
        }
        if !dir.pop() {
            // Fallback: use cwd/diagnose
            return std::env::current_dir().expect("cwd").join("diagnose");
        }
    }
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .init();

    let url = std::env::args().nth(1).expect("Usage: diagnose <URL>");
    let dir = out_dir();
    std::fs::create_dir_all(&dir).expect("Failed to create diagnose/ dir");
    eprintln!("Output dir: {}", dir.display());

    // ── Launch Chrome ──
    let mut builder = BrowserConfig::builder()
        .disable_default_args()
        .viewport(None)
        .with_head()
        .no_sandbox()
        .arg("--no-first-run")
        .arg("--no-default-browser-check")
        .arg("--disable-sync")
        .arg("--lang=en_US")
        .arg("--window-size=1920,1080")
        .arg("--remote-debugging-port=0");

    if let Ok(path) = std::env::var("CHROME_PATH") {
        builder = builder.chrome_executable(path);
    } else {
        for candidate in &[
            "/snap/chromium/current/usr/lib/chromium-browser/chrome",
            "/usr/bin/google-chrome-stable",
            "/usr/bin/google-chrome",
            "/usr/bin/chromium-browser",
            "/usr/bin/chromium",
        ] {
            if std::path::Path::new(candidate).exists() {
                builder = builder.chrome_executable(*candidate);
                break;
            }
        }
    }

    let config = builder.build().expect("Failed to build browser config");
    let (browser, mut handler) = Browser::launch(config).await.expect("Failed to launch Chrome");
    tokio::spawn(async move { while let Some(_) = handler.next().await {} });

    let page = browser.new_page("about:blank").await.expect("Failed to create page");

    // ── Navigate + wait for SPA ──
    eprintln!("[1/5] Navigating to: {}", url);
    let escaped = url.replace('\\', "\\\\").replace('\'', "\\'");
    let _ = page.evaluate(format!("window.location.assign('{escaped}')")).await;

    loop {
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        if let Ok(r) = page.evaluate("document.readyState").await {
            if let Ok(s) = r.into_value::<String>() {
                if s == "complete" { break; }
            }
        }
    }
    eprintln!("[1/5] readyState=complete, waiting 3s for SPA rendering...");
    tokio::time::sleep(std::time::Duration::from_secs(3)).await;

    // ── 1. Raw HTML ──
    let html = page.content().await.expect("Failed to get content");
    let raw_path = dir.join("1_raw.html");
    std::fs::write(&raw_path, &html).expect("write raw");
    eprintln!("[1/5] Raw HTML: {} bytes → {}", html.len(), raw_path.display());

    // ── 2. Cleaned HTML ──
    let cleaned = pipeline::clean_html(&html);
    let cleaned_path = dir.join("2_cleaned.html");
    std::fs::write(&cleaned_path, &cleaned).expect("write cleaned");
    eprintln!("[2/5] Cleaned HTML: {} bytes ({:.0}% reduction) → {}",
        cleaned.len(),
        (1.0 - cleaned.len() as f64 / html.len() as f64) * 100.0,
        cleaned_path.display());

    // Body text check
    let doc = Document::from(cleaned.as_str());
    let body_text = doc.select("body").text().to_string();
    let words: Vec<&str> = body_text.split_whitespace().collect();
    eprintln!("    Body text: {} words", words.len());
    for kw in &["$", "bedroom", "bed", "rating", "Superhost"] {
        let n = body_text.matches(kw).count();
        if n > 0 { eprintln!("    '{}' × {}", kw, n); }
        else { eprintln!("    '{}' NOT FOUND", kw); }
    }

    // ── 3. Chunk tree ──
    let chunks = pipeline::chunk_dom(&doc, &url);
    let total = pipeline::count_chunks(&chunks);
    let chunks_path = dir.join("3_chunks.txt");
    {
        let mut f = std::fs::File::create(&chunks_path).expect("create chunks");
        writeln!(f, "URL: {}", url).ok();
        writeln!(f, "{} chunks total\n", total).ok();

        let mut id_counter: usize = 0;
        fn write_chunk(f: &mut std::fs::File, c: &pipeline::DomChunk, depth: usize, counter: &mut usize) {
            *counter += 1;
            let id = *counter;
            let pad = "  ".repeat(depth);

            // Build a short label: tag + any useful attr
            let d = Document::from(c.html.as_str());
            let root_sel = d.select(&c.tag);
            let role = root_sel.attr("role").map(|v| v.to_string()).unwrap_or_default();
            let aria = root_sel.attr("aria-label").map(|v| v.to_string()).unwrap_or_default();
            let elem_id = root_sel.attr("id").map(|v| v.to_string()).unwrap_or_default();

            let label = if !elem_id.is_empty() {
                format!("{}#{}", c.tag, elem_id)
            } else if !role.is_empty() {
                format!("{}[{}]", c.tag, role)
            } else if !aria.is_empty() {
                let short_aria = if aria.len() > 40 { format!("{}...", safe_truncate(&aria, 40)) } else { aria };
                format!("{}[{}]", c.tag, short_aria)
            } else {
                c.tag.clone()
            };

            // Text content
            let text: String = d.select("body").text().to_string()
                .split_whitespace().collect::<Vec<_>>().join(" ");
            let text = if text.is_empty() {
                d.root().text().to_string()
                    .split_whitespace().collect::<Vec<_>>().join(" ")
            } else { text };
            let text_short = if text.len() > 150 {
                format!("{}...", safe_truncate(&text, 150))
            } else { text };

            // Actions
            let mut actions = vec![];
            let a_count = c.html.matches("<a ").count() + c.html.matches("<a>").count();
            let btn_count = c.html.matches("<button").count();
            let input_count = c.html.matches("<input").count();
            if a_count > 0 { actions.push(format!("{}link", a_count)); }
            if btn_count > 0 { actions.push(format!("{}btn", btn_count)); }
            if input_count > 0 { actions.push(format!("{}input", input_count)); }
            let action_str = if actions.is_empty() { String::new() }
                else { format!(" ({})", actions.join(" ")) };

            // Fold hint
            let fold_str = if !c.children.is_empty() {
                format!(" ▸ {} children", c.children.len())
            } else { String::new() };

            if text_short.is_empty() {
                writeln!(f, "{pad}#{id} {label}{action_str}{fold_str}").ok();
            } else {
                writeln!(f, "{pad}#{id} {label}{action_str}{fold_str} — {text_short}").ok();
            }

            for child in &c.children {
                write_chunk(f, child, depth + 1, counter);
            }
        }
        for chunk in &chunks {
            write_chunk(&mut f, chunk, 0, &mut id_counter);
        }
    }
    eprintln!("[3/5] Chunk tree: {} total → {}", total, chunks_path.display());

    // ── 4. Full LLM compression ──
    eprintln!("[4/5] Running full LLM compression pipeline...");
    dotenvy::dotenv().ok();
    let api_key = std::env::var("OPENROUTER_API_KEY").expect("OPENROUTER_API_KEY must be set");
    let model = std::env::var("WEBFURL_COMPRESSION_MODEL")
        .unwrap_or_else(|_| "openai/gpt-4o-mini".to_string());
    let generator = minillmlib::GeneratorInfo::openrouter(&model)
        .with_api_key(&api_key);
    let embedding_client = webfurl_core::embeddings::EmbeddingClient::new(api_key);
    let pipeline_config = pipeline::PipelineConfig {
        generator,
        embedding_client,
        max_depth: 6,
        min_content_length: 50,
    };

    // Connect to MongoDB for chunk-level caching
    let mongo_uri = std::env::var("MONGODB_URI").unwrap_or_else(|_| "mongodb://localhost:27017".to_string());
    let cache = webfurl_core::cache::CacheStore::new(&mongo_uri).await
        .expect("Failed to connect to MongoDB for cache");
    let result = pipeline::html_to_semantic_tree_cached(&html, &url, &pipeline_config, &cache).await;
    match result {
        Ok((tree, stats)) => {
            // Write fully unfolded tree
            let tree_path = dir.join("4_tree.txt");
            {
                let mut f = std::fs::File::create(&tree_path).expect("create tree");
                writeln!(f, "URL: {}", tree.url).ok();
                writeln!(f, "Title: {}", tree.title).ok();
                writeln!(f, "Compressed tokens: {}", tree.compressed_token_count).ok();
                writeln!(f, "Full tokens: {}", tree.full_token_count).ok();
                writeln!(f, "Root nodes: {}\n", tree.root_nodes.len()).ok();
                fn write_node(f: &mut std::fs::File, node: &SemanticNode, depth: usize) {
                    let pad = "  ".repeat(depth);
                    let actions: Vec<&str> = node.actions.iter().map(|a| match a {
                        webfurl_core::Action::Click { .. } => "click",
                        webfurl_core::Action::Fill { .. } => "fill",
                        webfurl_core::Action::Select { .. } => "select",
                        webfurl_core::Action::Toggle { .. } => "toggle",
                    }).collect();
                    let action_str = if actions.is_empty() { String::new() }
                        else { format!(" ({})", actions.join(",")) };
                    let dyn_str = if node.is_dynamic { " *dynamic*" } else { "" };
                    let stable_str = if node.stable { " [stable]" } else { "" };
                    writeln!(f, "{pad}[{summary}] {{#{id}}}{action_str}{dyn_str}{stable_str} tokens={t}",
                        summary = node.summary,
                        id = node.id,
                        t = node.token_count,
                    ).ok();
                    if let Some(raw) = &node.raw_text {
                        let raw_short = if raw.len() > 300 {
                            format!("{}...", safe_truncate(raw, 300))
                        } else { raw.clone() };
                        writeln!(f, "{pad}  [raw] {raw_short}").ok();
                    }
                    if !node.images.is_empty() {
                        writeln!(f, "{pad}  [{} images]", node.images.len()).ok();
                    }
                    for child in &node.children {
                        write_node(f, child, depth + 1);
                    }
                }
                for node in &tree.root_nodes {
                    write_node(&mut f, node, 0);
                }
            }
            eprintln!("[4/5] Semantic tree → {}", tree_path.display());

            // Write stats
            let stats_path = dir.join("5_stats.txt");
            {
                let mut f = std::fs::File::create(&stats_path).expect("create stats");
                writeln!(f, "URL: {}", url).ok();
                writeln!(f, "Raw HTML: {} bytes ({:.0} KB)", stats.raw_html_bytes, stats.raw_html_bytes as f64 / 1024.0).ok();
                writeln!(f, "Cleaned HTML: {} bytes ({:.0} KB)", stats.clean_html_bytes, stats.clean_html_bytes as f64 / 1024.0).ok();
                writeln!(f, "Reduction: {:.0}%", (1.0 - stats.clean_html_bytes as f64 / stats.raw_html_bytes as f64) * 100.0).ok();
                writeln!(f, "Total chunks: {}", stats.total_chunks).ok();
                writeln!(f, "Chunks cached: {}", stats.chunks_cached).ok();
                writeln!(f, "Chunks LLM-compressed: {}", stats.chunks_llm_compressed).ok();
                writeln!(f, "Compressed tokens: {}", stats.compressed_tokens).ok();
                writeln!(f, "Full tree tokens: {}", stats.full_tree_tokens).ok();
                writeln!(f, "Compression ratio: {:.2}", stats.compression_ratio()).ok();
                writeln!(f, "Duration: {}ms", stats.duration_ms).ok();
            }
            eprintln!("[5/5] Stats → {}", stats_path.display());
        }
        Err(e) => {
            eprintln!("[4/5] COMPRESSION FAILED: {e}");
            let err_path = dir.join("4_error.txt");
            std::fs::write(&err_path, format!("{e:?}")).ok();
        }
    }

    eprintln!("\n=== Done. All files in {} ===", dir.display());
}

/// Truncate a string at a valid UTF-8 char boundary.
fn safe_truncate(s: &str, max_len: usize) -> &str {
    if s.len() <= max_len {
        return s;
    }
    let mut end = max_len;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}
