use minillmlib::{ChatNode, GeneratorInfo, CompletionParameters, NodeCompletionParameters};
use std::io::{self, Write};
use std::sync::Arc;
use tracing::info;

use crate::browser::BrowserSession;

use webfurl_core::{
    cache::CacheStore,
    debug as wf_debug,
    embeddings::EmbeddingClient,
    pipeline::{self, CompressionRunStats, PipelineConfig},
    serialize::{self, CollapsedPage},
    unfold::{self, UnfoldState},
    tree::SemanticTree,
    vision::VisionClient,
};

const SYSTEM_PROMPT: &str = "You are a web browsing agent. You control a real Chrome browser. You see pages through a compressed semantic tree. To act, output a single JSON object. To answer the user, output plain text. Be concise.";

const WEBFURL_INSTRUCTIONS: &str = r#"## How to read the page tree below

Each node: [Summary] {#node_id} (interactivity)

Markers:
- {#node_id} — the node's unique ID (use this in actions)
- (clickable) / (fillable) / (selectable) / (toggleable) — what you can do with this element
- ... (N children, +M tokens to unfold) — COLLAPSED CHILDREN you can UNFOLD

## UNFOLD vs CLICK — do NOT confuse them

UNFOLD = expand a collapsed tree node. ONLY works when you see "... (N children, +M tokens to unfold)". Does NOT touch the browser.
CLICK = browser interaction. ONLY works on nodes marked (clickable). Performs a real click and reloads the page.
No interactivity marker and no fold hint = leaf node, no further action possible.

## Available actions (output ONE JSON object, then STOP)

{"action":"web_search","query":"search terms"} — Google search, opens results page
{"action":"navigate","url":"https://example.com"} — go to a specific URL
{"action":"unfold","node_id":"id"} — ONLY if "... (N children, +M tokens to unfold)" is shown
{"action":"page_search","query":"terms"} — semantic search within current page (requires loaded page)
{"action":"click","node_id":"id"} — ONLY if node is marked (clickable)
{"action":"fill","node_id":"id","text":"value"} — ONLY if node is marked (fillable)
{"action":"describe","node_id":"id"} — get image descriptions for a node
{"action":"ask_image","node_id":"id","question":"what is shown?"}

After click/fill/navigate, page is re-read and fresh tree provided automatically.

## Rules
1. To act: ONE JSON object, then STOP. Nothing after the JSON.
2. To answer: plain text only. No JSON.
3. Never output [WEBFURL] tags. Never predict results."#;

/// Cumulative session-level stats, updated on every pipeline run.
struct SessionStats {
    total_compressions: u32,
    total_page_cache_hits: u32,
    total_chunks_cached: u32,
    total_chunks_llm: u32,
    total_raw_html_bytes: usize,
    total_clean_html_bytes: usize,
    total_compressed_tokens: u64,
    total_full_tree_tokens: u64,
    total_duration_ms: u64,
    /// Current page's last run stats
    last_run: Option<CompressionRunStats>,
    /// Current unfold budget usage
    current_budget_used: u32,
    current_budget_max: u32,
    /// Number of agent LLM turns this session
    agent_turns: u32,
}

impl SessionStats {
    fn new() -> Self {
        Self {
            total_compressions: 0,
            total_page_cache_hits: 0,
            total_chunks_cached: 0,
            total_chunks_llm: 0,
            total_raw_html_bytes: 0,
            total_clean_html_bytes: 0,
            total_compressed_tokens: 0,
            total_full_tree_tokens: 0,
            total_duration_ms: 0,
            last_run: None,
            current_budget_used: 0,
            current_budget_max: 0,
            agent_turns: 0,
        }
    }

    fn record_run(&mut self, run: &CompressionRunStats) {
        self.total_compressions += 1;
        if run.page_cache_hit { self.total_page_cache_hits += 1; }
        self.total_chunks_cached += run.chunks_cached;
        self.total_chunks_llm += run.chunks_llm_compressed;
        self.total_raw_html_bytes += run.raw_html_bytes;
        self.total_clean_html_bytes += run.clean_html_bytes;
        self.total_compressed_tokens += run.compressed_tokens as u64;
        self.total_full_tree_tokens += run.full_tree_tokens as u64;
        self.total_duration_ms += run.duration_ms;
        self.last_run = Some(run.clone());
    }

    fn format_bytes(bytes: usize) -> String {
        if bytes < 1024 { return format!("{bytes} B"); }
        if bytes < 1024 * 1024 { return format!("{:.1} KB", bytes as f64 / 1024.0); }
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    }

    fn to_markdown(&self) -> String {
        let mut out = String::from("# Webfurl Live Stats\n\n");

        // Session overview
        out.push_str("## Session Overview\n\n");
        out.push_str(&format!("| Metric | Value |\n|--------|-------|\n"));
        out.push_str(&format!("| Pipeline runs | {} |\n", self.total_compressions));
        out.push_str(&format!("| Page cache hits | {} |\n", self.total_page_cache_hits));
        out.push_str(&format!("| Chunks cached (subtree) | {} |\n", self.total_chunks_cached));
        out.push_str(&format!("| Chunks LLM-compressed | {} |\n", self.total_chunks_llm));
        out.push_str(&format!("| Agent LLM turns | {} |\n", self.agent_turns));
        out.push_str(&format!("| Total pipeline time | {:.1}s |\n", self.total_duration_ms as f64 / 1000.0));
        out.push_str("\n");

        // Token savings
        out.push_str("## Token Savings\n\n");
        let est_raw = self.total_clean_html_bytes / 4;
        let compressed = self.total_compressed_tokens;
        let saved = est_raw as i64 - compressed as i64;
        let ratio = if est_raw > 0 { compressed as f64 / est_raw as f64 } else { 0.0 };
        out.push_str(&format!("| Metric | Value |\n|--------|-------|\n"));
        out.push_str(&format!("| Raw HTML (all runs) | {} |\n", Self::format_bytes(self.total_raw_html_bytes)));
        out.push_str(&format!("| Clean HTML (all runs) | {} |\n", Self::format_bytes(self.total_clean_html_bytes)));
        out.push_str(&format!("| Est. raw tokens (clean/4) | ~{} |\n", est_raw));
        out.push_str(&format!("| Compressed tokens (all runs) | {} |\n", compressed));
        out.push_str(&format!("| Tokens saved (cumulative) | **{}** |\n", saved));
        out.push_str(&format!("| Compression ratio | **{:.1}%** |\n", ratio * 100.0));
        out.push_str("\n");

        // Current page budget
        out.push_str("## Current Page Budget\n\n");
        out.push_str(&format!("| Metric | Value |\n|--------|-------|\n"));
        out.push_str(&format!("| Budget used | {} tokens |\n", self.current_budget_used));
        out.push_str(&format!("| Budget max | {} tokens |\n", self.current_budget_max));
        let budget_pct = if self.current_budget_max > 0 {
            self.current_budget_used as f64 / self.current_budget_max as f64 * 100.0
        } else { 0.0 };
        out.push_str(&format!("| Budget utilization | {:.0}% |\n", budget_pct));
        out.push_str("\n");

        // Last compression run
        if let Some(run) = &self.last_run {
            out.push_str("## Last Compression Run\n\n");
            out.push_str(&format!("| Metric | Value |\n|--------|-------|\n"));
            out.push_str(&format!("| Page cache hit | {} |\n", if run.page_cache_hit { "yes" } else { "no" }));
            out.push_str(&format!("| Raw HTML | {} |\n", Self::format_bytes(run.raw_html_bytes)));
            out.push_str(&format!("| Clean HTML | {} |\n", Self::format_bytes(run.clean_html_bytes)));
            out.push_str(&format!("| Total chunks | {} |\n", run.total_chunks));
            out.push_str(&format!("| Chunks cached | {} |\n", run.chunks_cached));
            out.push_str(&format!("| Chunks LLM-compressed | {} |\n", run.chunks_llm_compressed));
            let est = run.estimated_raw_tokens();
            out.push_str(&format!("| Est. raw tokens | ~{} |\n", est));
            out.push_str(&format!("| Compressed tokens | {} |\n", run.compressed_tokens));
            out.push_str(&format!("| Full tree tokens | {} |\n", run.full_tree_tokens));
            let saved = run.tokens_saved_vs_raw();
            out.push_str(&format!("| Tokens saved | **{}** |\n", saved));
            out.push_str(&format!("| Compression ratio | {:.1}% |\n", run.compression_ratio() * 100.0));
            out.push_str(&format!("| Duration | {}ms |\n", run.duration_ms));
        }

        out
    }
}

pub struct AgentContext {
    agent_generator: GeneratorInfo,
    pipeline_config: PipelineConfig,
    embedding_client: EmbeddingClient,
    vision_client: VisionClient,
    cache: CacheStore,
    browser: BrowserSession,
    /// Per-page initial furl budget (auto-unfold fills up to this)
    initial_budget: u32,
    /// Hard ceiling across the session the agent can expand to
    max_budget: u32,

    // Conversation state
    #[allow(dead_code)]
    conversation_root: Arc<ChatNode>,
    conversation_tip: Arc<ChatNode>,
    /// Tracks the ephemeral WEBFURL context node so we can detach it after use.
    current_context_node: Option<Arc<ChatNode>>,
    /// Snapshot of the last thread sent to the LLM (includes WEBFURL).
    /// Used for the context file — always shows what the LLM actually saw.
    last_llm_thread: Vec<minillmlib::message::Message>,

    // Page state
    current_tree: Option<SemanticTree>,
    unfold_state: Option<UnfoldState>,
    collapsed_pages: Vec<CollapsedPage>,

    /// Path to the live context file, updated on every state change.
    context_file_path: std::path::PathBuf,
    /// Path to the live stats file.
    stats_file_path: std::path::PathBuf,
    /// Session-level cumulative stats.
    session_stats: SessionStats,
    /// True when a CAPTCHA was detected and we're waiting for the user to solve it.
    pending_captcha: bool,
}

impl AgentContext {
    pub fn new(
        agent_generator: GeneratorInfo,
        pipeline_config: PipelineConfig,
        embedding_client: EmbeddingClient,
        cache: CacheStore,
        initial_budget: u32,
        max_budget: u32,
        browser: BrowserSession,
        vision_generator: GeneratorInfo,
    ) -> Self {
        let vision_client = VisionClient::new(
            vision_generator,
            embedding_client.clone(),
            cache.clone(),
        );
        let root = ChatNode::root(SYSTEM_PROMPT);
        let tip = root.clone();
        let initial_thread = tip.thread();
        let context_file_path = std::path::PathBuf::from("webfurl_context.md");
        let stats_file_path = std::path::PathBuf::from("webfurl_stats.md");
        // Create the file immediately with the system prompt visible
        {
            let mut init = String::new();
            for msg in &initial_thread {
                let role_label = match msg.role {
                    minillmlib::message::Role::System => "**[system]**",
                    minillmlib::message::Role::User => "**[user]**",
                    minillmlib::message::Role::Assistant => "**[assistant]**",
                    _ => "[other]",
                };
                let content = msg.content.get_text().unwrap_or("");
                init.push_str(&format!("{role_label}\n{content}\n\n"));
            }
            init.push_str("_Waiting for input..._\n");
            let _ = std::fs::write(&context_file_path, &init);
        }

        Self {
            agent_generator,
            pipeline_config,
            embedding_client,
            vision_client,
            cache,
            browser,
            initial_budget,
            max_budget,
            conversation_root: root,
            conversation_tip: tip,
            current_context_node: None,
            last_llm_thread: initial_thread,
            current_tree: None,
            unfold_state: None,
            collapsed_pages: vec![],
            context_file_path,
            stats_file_path,
            session_stats: SessionStats::new(),
            pending_captcha: false,
        }
    }

    pub async fn handle_input(&mut self, input: &str) -> Result<String, String> {
        if let Some(url) = input.strip_prefix("/url ") {
            return self.navigate(url.trim()).await;
        }
        if let Some(node_id) = input.strip_prefix("/unfold ") {
            return self.manual_unfold(node_id.trim());
        }
        if let Some(node_id) = input.strip_prefix("/fold ") {
            return self.manual_fold(node_id.trim());
        }
        if let Some(query) = input.strip_prefix("/search ") {
            return self.manual_search(query.trim()).await;
        }
        if input == "/tree" {
            return self.show_tree();
        }
        if input == "/screenshot" {
            return self.screenshot_page().await;
        }
        if let Some(selector) = input.strip_prefix("/screenshot ") {
            return self.screenshot_element(selector.trim()).await;
        }
        if let Some(node_id) = input.strip_prefix("/describe ") {
            return self.describe_images(node_id.trim()).await;
        }
        if let Some(rest) = input.strip_prefix("/ask_image ") {
            let parts: Vec<&str> = rest.splitn(2, ' ').collect();
            if parts.len() < 2 {
                return Err("Usage: /ask_image <node_id> <question>".into());
            }
            return self.ask_image(parts[0].trim(), parts[1].trim()).await;
        }
        if let Some(selector) = input.strip_prefix("/click ") {
            return self.execute_click(selector.trim()).await;
        }
        if let Some(rest) = input.strip_prefix("/fill ") {
            let parts: Vec<&str> = rest.splitn(2, ' ').collect();
            if parts.len() < 2 {
                return Err("Usage: /fill <css_selector> <text>".into());
            }
            return self.execute_fill(parts[0].trim(), parts[1].trim()).await;
        }

        if input == "/browser" {
            return self.browser.open_in_browser().await;
        }

        // If a CAPTCHA was pending, any user message means they solved it.
        // Re-read the page before proceeding.
        if self.pending_captcha {
            self.pending_captcha = false;
            info!("user indicated CAPTCHA resolved — re-reading page");

            // Brief wait for page to settle after CAPTCHA solve
            tokio::time::sleep(std::time::Duration::from_millis(1000)).await;

            // Check if CAPTCHA is actually gone
            if let Ok(Some(still)) = self.browser.detect_captcha().await {
                self.pending_captcha = true;
                return Ok(format!(
                    "CAPTCHA still detected: {still}\n\
                     Please finish solving it in the browser window, then let me know."
                ));
            }

            // CAPTCHA gone — re-read and compress the page
            self.refresh_page().await.map_err(|e| format!("Post-CAPTCHA refresh failed: {e}"))?;
            info!("page refreshed after CAPTCHA — continuing");
        }

        // Sync browser state: detect if user changed the page (navigation, form fill, click, etc.)
        self.sync_browser_state().await;

        // Regular user message → send to agent with current page context
        self.agent_turn(input).await
    }

    async fn navigate(&mut self, url: &str) -> Result<String, String> {
        info!(url, "navigating to URL");

        // Collapse current page if we have one
        if let Some(tree) = &self.current_tree {
            let collapsed = serialize::collapse_tree(tree, "");
            // Deduplicate: replace existing entry for same domain, don't accumulate duplicates
            if let Some(existing) = self.collapsed_pages.iter_mut().find(|p| p.domain == collapsed.domain) {
                *existing = collapsed;
            } else {
                self.collapsed_pages.push(collapsed);
            }
        }

        // Navigate browser and get rendered HTML
        let html = self.browser.navigate(url).await?;

        // Check for CAPTCHA before compressing
        if let Ok(Some(captcha_type)) = self.browser.detect_captcha().await {
            info!(captcha = %captcha_type, "CAPTCHA detected — waiting for user");
            self.pending_captcha = true;
            return Ok(format!(
                "🔒 CAPTCHA detected: {captcha_type}\n\
                 The browser window is showing the challenge.\n\
                 Please solve it, then type /done to continue."
            ));
        }

        // Run through the pipeline (with chunk-level content-hash cache)
        let (tree, run_stats) = pipeline::html_to_semantic_tree_cached(&html, url, &self.pipeline_config, &self.cache)
            .await
            .map_err(|e| format!("Pipeline error: {e}"))?;

        let state = unfold::initial_pack(&tree, self.initial_budget, self.max_budget);

        // Auto-unfold top-level nodes to fill the budget
        let mut state = state;
        let auto_unfolded = unfold::auto_unfold(&tree, &mut state);

        info!(
            compressed = tree.compressed_token_count,
            full = tree.full_token_count,
            ratio = %format!("{:.1}%", tree.compression_ratio() * 100.0),
            budget_used = state.token_usage,
            auto_unfolded = auto_unfolded.len(),
            "page compressed"
        );

        // Rich debug visualization
        wf_debug::print_compression_stats(&tree);
        wf_debug::print_tree(&tree);
        wf_debug::print_unfold_state(&state, &tree);
        wf_debug::print_context_window(&tree, &state, &self.collapsed_pages, None);

        // Update stats
        self.session_stats.record_run(&run_stats);
        self.session_stats.current_budget_used = state.token_usage;
        self.session_stats.current_budget_max = self.max_budget;

        self.current_tree = Some(tree);
        self.unfold_state = Some(state);
        self.dump_context_file();
        self.dump_stats_file();

        Ok(format!("Navigated to {url}. Type a question or use /unfold, /search, /tree."))
    }

    async fn screenshot_page(&self) -> Result<String, String> {
        let png = self.browser.screenshot_page().await?;
        let path = "/tmp/webfurl_screenshot.png";
        std::fs::write(path, &png).map_err(|e| format!("Failed to write screenshot: {e}"))?;
        Ok(format!("Screenshot saved to {path} ({} bytes)", png.len()))
    }

    async fn screenshot_element(&self, selector: &str) -> Result<String, String> {
        let png = self.browser.screenshot_element(selector).await?;
        let path = format!("/tmp/webfurl_screenshot_{}.png", selector.replace(['.', '#', ' ', '>', '[', ']'], "_"));
        std::fs::write(&path, &png).map_err(|e| format!("Failed to write screenshot: {e}"))?;
        Ok(format!("Element screenshot saved to {path} ({} bytes)", png.len()))
    }

    /// Resolve a node_id to the CSS selector of its first Click action.
    fn resolve_click_selector(&self, node_id: &str) -> Result<String, String> {
        let tree = self.current_tree.as_ref().ok_or("No page loaded")?;
        let node = tree.find_node(node_id)
            .ok_or_else(|| format!("Node #{node_id} not found in tree"))?;
        for action in &node.actions {
            if let webfurl_core::actions::Action::Click { selector, .. } = action {
                return Ok(selector.clone());
            }
        }
        Err(format!("Node #{node_id} has no click action"))
    }

    /// Resolve a node_id to the CSS selector of its first Fill action.
    fn resolve_fill_selector(&self, node_id: &str) -> Result<String, String> {
        let tree = self.current_tree.as_ref().ok_or("No page loaded")?;
        let node = tree.find_node(node_id)
            .ok_or_else(|| format!("Node #{node_id} not found in tree"))?;
        for action in &node.actions {
            if let webfurl_core::actions::Action::Fill { selector, .. } = action {
                return Ok(selector.clone());
            }
        }
        Err(format!("Node #{node_id} has no fill action"))
    }

    async fn execute_click(&mut self, selector: &str) -> Result<String, String> {
        info!(selector, "executing CLICK");
        let url_before = self.browser.current_url().await.unwrap_or_default();
        let tab_count_before = self.browser.page_count().await;

        self.browser.click(selector).await?;

        // Brief pause to let click propagate (navigation or new tab)
        tokio::time::sleep(std::time::Duration::from_millis(300)).await;

        // Check if click opened a new tab (target="_blank")
        let switched = self.browser.switch_to_newest_tab(tab_count_before).await;
        if switched {
            // New tab opened — wait for it to load
            self.browser.wait_for_page_stable().await?;
        } else {
            // Same tab — check if URL changed (same-tab navigation)
            let url_after = self.browser.current_url().await.unwrap_or_default();
            if url_after != url_before {
                info!(from = %url_before, to = %url_after, "click triggered navigation — waiting for page to load");
                self.browser.wait_for_page_stable().await?;
            } else {
                // SPA-style click (modal, tab switch, etc.) — wait briefly for DOM update
                tokio::time::sleep(std::time::Duration::from_millis(500)).await;
            }
        }

        // Re-read DOM and update tree
        self.refresh_page().await?;
        Ok(format!("Clicked '{selector}'. Page re-read and tree updated."))
    }

    async fn execute_fill(&mut self, selector: &str, text: &str) -> Result<String, String> {
        info!(selector, text_len = text.len(), "executing FILL");
        self.browser.fill(selector, text).await?;

        tokio::time::sleep(std::time::Duration::from_millis(300)).await;

        // Re-read DOM and update tree
        self.refresh_page().await?;
        Ok(format!("Typed into '{selector}'. Page re-read and tree updated."))
    }

    /// Re-read the current page DOM and recompress the tree.
    /// Always runs fresh LLM compression — after click/fill the DOM state has changed
    /// even if the structural hash looks the same (SPA class toggles, React state, etc.).
    async fn refresh_page(&mut self) -> Result<(), String> {
        let url = self.browser.current_url().await?;
        info!(url = %url, "refreshing page after action (no cache — post-interaction)");

        // Check for CAPTCHA after interaction
        if let Ok(Some(captcha_type)) = self.browser.detect_captcha().await {
            info!(captcha = %captcha_type, "CAPTCHA detected after action — waiting for user");
            self.pending_captcha = true;
            return Err(format!(
                "🔒 CAPTCHA detected: {captcha_type}\n\
                 The browser window is showing the challenge.\n\
                 Please solve it, then type /done to continue."
            ));
        }

        let html = self.browser.get_current_page_content().await?;

        let (tree, run_stats) = pipeline::html_to_semantic_tree_cached(&html, &url, &self.pipeline_config, &self.cache)
            .await
            .map_err(|e| format!("Pipeline error on refresh: {e}"))?;

        // Preserve budget usage but reset unfold state
        // (the tree structure may have changed, so old unfold IDs may not exist)
        let state = unfold::initial_pack(&tree, self.initial_budget, self.max_budget);
        let mut state = state;
        unfold::auto_unfold(&tree, &mut state);

        info!(
            compressed = tree.compressed_token_count,
            budget_used = state.token_usage,
            "page refreshed after action"
        );

        wf_debug::print_compression_stats(&tree);
        wf_debug::print_tree(&tree);
        wf_debug::print_unfold_state(&state, &tree);

        // Update stats
        self.session_stats.record_run(&run_stats);
        self.session_stats.current_budget_used = state.token_usage;
        self.session_stats.current_budget_max = self.max_budget;

        self.current_tree = Some(tree);
        self.unfold_state = Some(state);
        self.dump_context_file();
        self.dump_stats_file();

        Ok(())
    }

    /// Detect if the user changed the browser state (navigated, clicked, filled forms, etc.)
    /// by comparing the current DOM structural hash to the last known one.
    /// If changed, recompresses the page (with cache — unchanged subtrees are fast).
    async fn sync_browser_state(&mut self) {
        // Nothing to sync if no page has been loaded yet
        if self.current_tree.is_none() {
            // But check if user navigated somewhere on their own
            if let Ok(url) = self.browser.current_url().await {
                if url != "about:blank" {
                    info!(url = %url, "user navigated to a page — loading it");
                    let _ = self.load_current_page().await;
                }
            }
            return;
        }

        let current_hash = self.current_tree.as_ref().unwrap().structural_hash.clone();
        let current_url = self.current_tree.as_ref().unwrap().url.clone();

        // Check URL first — did user navigate away entirely?
        let browser_url = match self.browser.current_url().await {
            Ok(u) => u,
            Err(_) => return,
        };

        if browser_url != current_url {
            info!(
                old = %current_url,
                new = %browser_url,
                "user navigated to a different page — reloading"
            );
            let _ = self.load_current_page().await;
            return;
        }

        // Same URL — check if DOM structure changed (user interaction, lazy load, etc.)
        let html = match self.browser.get_current_page_content().await {
            Ok(h) => h,
            Err(_) => return,
        };

        let new_hash = pipeline::structural_hash_of_html(&html);
        if new_hash == current_hash {
            return;
        }

        info!(
            old_hash = %&current_hash[..8],
            new_hash = %&new_hash[..8],
            "DOM changed since last read — recompressing"
        );

        // Recompress with chunk-level cache
        match pipeline::html_to_semantic_tree_cached(&html, &browser_url, &self.pipeline_config, &self.cache).await {
            Ok((tree, run_stats)) => {
                let state = unfold::initial_pack(&tree, self.initial_budget, self.max_budget);
                let mut state = state;
                unfold::auto_unfold(&tree, &mut state);

                self.session_stats.record_run(&run_stats);
                self.session_stats.current_budget_used = state.token_usage;
                self.session_stats.current_budget_max = self.max_budget;

                self.current_tree = Some(tree);
                self.unfold_state = Some(state);
                self.dump_context_file();
                self.dump_stats_file();

                info!(
                    cached = run_stats.chunks_cached,
                    llm = run_stats.chunks_llm_compressed,
                    "page recompressed after user change"
                );
            }
            Err(e) => {
                info!("recompression after user change failed: {e}");
            }
        }
    }

    /// Load whatever page is currently in the browser (used when user navigated on their own).
    async fn load_current_page(&mut self) -> Result<(), String> {
        let url = self.browser.current_url().await?;
        let html = self.browser.get_current_page_content().await?;

        // Check for CAPTCHA
        if let Ok(Some(captcha_type)) = self.browser.detect_captcha().await {
            info!(captcha = %captcha_type, "CAPTCHA on user-navigated page");
            self.pending_captcha = true;
            return Ok(());
        }

        let (tree, run_stats) = pipeline::html_to_semantic_tree_cached(&html, &url, &self.pipeline_config, &self.cache)
            .await
            .map_err(|e| format!("Pipeline error: {e}"))?;

        let state = unfold::initial_pack(&tree, self.initial_budget, self.max_budget);
        let mut state = state;
        unfold::auto_unfold(&tree, &mut state);

        self.session_stats.record_run(&run_stats);
        self.session_stats.current_budget_used = state.token_usage;
        self.session_stats.current_budget_max = self.max_budget;

        self.current_tree = Some(tree);
        self.unfold_state = Some(state);
        self.dump_context_file();
        self.dump_stats_file();

        Ok(())
    }

    pub async fn close_browser(self) -> Result<(), String> {
        self.browser.close().await
    }

    fn manual_unfold(&mut self, node_id: &str) -> Result<String, String> {
        let result = {
            let tree = self.current_tree.as_ref().ok_or("No page loaded")?;
            let state = self.unfold_state.as_mut().ok_or("No page loaded")?;

            match unfold::unfold_node(tree, state, node_id) {
                Some(0) => Ok(format!("Node #{node_id} is already unfolded or has no children")),
                Some(cost) => {
                    wf_debug::print_unfold_state(state, tree);
                    wf_debug::print_context_window(tree, state, &self.collapsed_pages, None);
                    Ok(format!(
                        "Unfolded #{node_id} (+{cost} tokens, {}/{} budget used)",
                        state.token_usage, self.max_budget
                    ))
                }
                None => Err(format!(
                    "Cannot unfold #{node_id}: would exceed budget ({}/{})",
                    state.token_usage, self.max_budget
                )),
            }
        };
        if result.is_ok() {
            if let Some(state) = &self.unfold_state {
                self.session_stats.current_budget_used = state.token_usage;
            }
            self.dump_context_file();
            self.dump_stats_file();
        }
        result
    }

    fn manual_fold(&mut self, node_id: &str) -> Result<String, String> {
        let result = {
            let tree = self.current_tree.as_ref().ok_or("No page loaded")?;
            let state = self.unfold_state.as_mut().ok_or("No page loaded")?;

            match unfold::fold_node(tree, state, node_id) {
                Some(reclaimed) => {
                    wf_debug::print_unfold_state(state, tree);
                    wf_debug::print_context_window(tree, state, &self.collapsed_pages, None);
                    Ok(format!(
                        "Folded #{node_id} (-{reclaimed} tokens, {}/{} budget used)",
                        state.token_usage, self.max_budget
                    ))
                }
                None => Err(format!("Node #{node_id} is not currently unfolded")),
            }
        };
        if result.is_ok() {
            if let Some(state) = &self.unfold_state {
                self.session_stats.current_budget_used = state.token_usage;
            }
            self.dump_context_file();
            self.dump_stats_file();
        }
        result
    }

    async fn manual_search(&mut self, query: &str) -> Result<String, String> {
        let tree = self.current_tree.as_ref().ok_or("No page loaded")?;
        let state = self.unfold_state.as_mut().ok_or("No page loaded")?;

        let query_embedding = self.embedding_client
            .embed(query)
            .await
            .map_err(|e| format!("Embedding error: {e}"))?;

        let unfolded = unfold::semantic_unfold(tree, state, &query_embedding, 5);

        wf_debug::print_unfold_state(state, tree);
        wf_debug::print_context_window(tree, state, &self.collapsed_pages, Some(query));

        let result = if unfolded.is_empty() {
            Ok(format!("No relevant nodes found for \"{query}\""))
        } else {
            Ok(format!(
                "Search \"{query}\" unfolded {} nodes: {}\nBudget: {}/{}",
                unfolded.len(),
                unfolded.join(", "),
                state.token_usage,
                self.max_budget,
            ))
        };
        if let Some(state) = &self.unfold_state {
            self.session_stats.current_budget_used = state.token_usage;
        }
        self.dump_context_file();
        self.dump_stats_file();
        result
    }

    async fn describe_images(&mut self, node_id: &str) -> Result<String, String> {
        let tree = self.current_tree.as_mut().ok_or("No page loaded")?;
        let node = tree.find_node_mut(node_id)
            .ok_or_else(|| format!("Node #{node_id} not found"))?;

        if node.images.is_empty() {
            return Ok(format!("Node #{node_id} has no images."));
        }

        let stats = self.vision_client
            .describe_all_images(&mut node.images)
            .await
            .map_err(|e| format!("Vision error: {e}"))?;

        let mut output = format!("Described images in #{node_id}: {stats}\n");
        for img in &node.images {
            if let Some(desc) = &img.description {
                output.push_str(&format!("  [img] {desc}\n"));
            }
        }
        Ok(output)
    }

    async fn ask_image(&mut self, node_id: &str, question: &str) -> Result<String, String> {
        let tree = self.current_tree.as_ref().ok_or("No page loaded")?;
        let node = tree.find_node(node_id)
            .ok_or_else(|| format!("Node #{node_id} not found"))?;

        if node.images.is_empty() {
            return Ok(format!("Node #{node_id} has no images."));
        }

        // Ask the question about the first image in the node
        let image = &node.images[0];
        let (answer, cache_hit) = self.vision_client
            .query_image(image, question)
            .await
            .map_err(|e| format!("Vision error: {e}"))?;

        let hit_str = if cache_hit { " (cache hit)" } else { " (computed)" };
        Ok(format!("Image in #{node_id}{hit_str}:\n{answer}"))
    }

    fn show_tree(&self) -> Result<String, String> {
        let tree = self.current_tree.as_ref().ok_or("No page loaded")?;
        let state = self.unfold_state.as_ref().ok_or("No page loaded")?;
        wf_debug::print_compression_stats(tree);
        wf_debug::print_tree(tree);
        wf_debug::print_unfold_state(state, tree);
        wf_debug::print_context_window(tree, state, &self.collapsed_pages, None);
        Ok("Tree displayed above.".to_string())
    }

    /// Dump the context file: just the conversation as the LLM sees it.
    /// If `streaming_suffix` is provided, appends it as a partial assistant message.
    fn dump_context_file_ex(&self, streaming_suffix: Option<&str>) {
        let mut out = String::new();

        for msg in &self.last_llm_thread {
            let role_label = match msg.role {
                minillmlib::message::Role::System => "**[system]**",
                minillmlib::message::Role::User => "**[user]**",
                minillmlib::message::Role::Assistant => "**[assistant]**",
                _ => "[other]",
            };
            let content = msg.content.get_text().unwrap_or("");
            out.push_str(&format!("{role_label}\n{content}\n\n"));
        }

        // Append partial streaming response if present
        if let Some(partial) = streaming_suffix {
            out.push_str(&format!("**[assistant]** _(streaming...)_\n{partial}\n\n"));
        }

        let _ = std::fs::write(&self.context_file_path, &out);
    }

    fn dump_context_file(&self) {
        self.dump_context_file_ex(None);
    }

    /// Write the live stats file (overwrites on every call).
    fn dump_stats_file(&self) {
        let md = self.session_stats.to_markdown();
        let _ = std::fs::write(&self.stats_file_path, &md);
    }

    /// Entry point for an agent turn. Handles the full cycle:
    /// user input → [WEBFURL context] → LLM completion → execute commands → follow-up turns.
    ///
    /// Conversation structure:
    ///   conversation_tip (last real exchange)
    ///     └── user: "user input"
    ///           └── user: "[WEBFURL]...[/WEBFURL]"   ← context node (detached & replaced on follow-ups)
    ///                 └── assistant: "response"
    async fn agent_turn(&mut self, user_input: &str) -> Result<String, String> {
        // Query-driven pre-unfolding: embed the user's query and unfold the most
        // relevant nodes so the LLM sees a page view focused on what the user asked about.
        if self.current_tree.is_some() && self.unfold_state.is_some() {
            if let Ok(query_emb) = self.embedding_client.embed(user_input).await {
                let tree = self.current_tree.as_ref().unwrap();
                let state = self.unfold_state.as_mut().unwrap();
                let pre_unfolded = unfold::semantic_unfold(tree, state, &query_emb, 5);
                if !pre_unfolded.is_empty() {
                    info!(
                        query = user_input,
                        unfolded = pre_unfolded.len(),
                        budget = state.token_usage,
                        "query-driven pre-unfold"
                    );
                }
            }
        }

        let user_anchor = self.conversation_tip.add_user(user_input);

        let final_assistant = self.completion_loop(&user_anchor, None, 0).await?;

        self.conversation_tip = final_assistant;
        self.dump_context_file();
        Ok(String::new())
    }

    /// Run one LLM completion with WEBFURL context, then process commands.
    /// If commands produce results, detach the old context node, create a new one
    /// with command results + fresh WEBFURL, and recurse.
    fn completion_loop<'a>(
        &'a mut self,
        user_anchor: &'a Arc<ChatNode>,
        command_preamble: Option<String>,
        depth: u32,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Arc<ChatNode>, String>> + 'a>> {
        Box::pin(async move {
            const MAX_DEPTH: u32 = 25;
            if depth >= MAX_DEPTH {
                info!(depth, "max agent follow-up depth reached");
                return Ok(user_anchor.add_assistant("(max follow-up depth reached)"));
            }

            // Build context: instructions always present + WEBFURL block (or empty browser state)
            let webfurl_block = if let (Some(tree), Some(state)) =
                (&self.current_tree, &self.unfold_state)
            {
                let tree_text = serialize::serialize_tree(tree, state, &self.collapsed_pages);
                format!("{WEBFURL_INSTRUCTIONS}\n\n{tree_text}")
            } else {
                format!("{WEBFURL_INSTRUCTIONS}\n\n[WEBFURL]\n[no page loaded — use navigate or web_search to open a page]\n[/WEBFURL]")
            };

            let context_content = match &command_preamble {
                Some(preamble) => format!("{webfurl_block}\n\n{preamble}"),
                None => webfurl_block,
            };

            // Detach old context node from previous iteration
            if let Some(old_ctx) = self.current_context_node.take() {
                old_ctx.detach();
            }

            let context_node = {
                let node = user_anchor.add_user(context_content.as_str());
                self.current_context_node = Some(node.clone());
                node
            };

            // Snapshot the thread (includes WEBFURL) for the context file
            self.last_llm_thread = context_node.thread();
            self.dump_context_file();

            // Track agent turns
            self.session_stats.agent_turns += 1;
            self.dump_stats_file();

            // Stream completion
            let params = NodeCompletionParameters::new()
                .with_params(
                    CompletionParameters::new()
                        .with_temperature(0.3)
                        .with_max_tokens(4096),
                );

            let mut stream = context_node
                .complete_streaming(&self.agent_generator, Some(&params))
                .await
                .map_err(|e| format!("Agent LLM error: {e}"))?;

            print!("\n");
            let raw_response = self.stream_to_terminal(&mut stream).await;
            println!();

            // Add assistant response to the tree (temporary — will be detached with context_node)
            let _assistant_node = context_node.add_assistant(raw_response.as_str());

            // Update snapshot with assistant response
            self.last_llm_thread.push(minillmlib::message::Message::assistant(raw_response.as_str()));
            let response_text = raw_response;
            self.dump_context_file();

            // Process commands — only recurse if at least one succeeded
            let (cmd_results, any_success) = self.process_agent_commands(&response_text).await;

            // Clean up: detach WEBFURL context node, re-attach assistant directly
            // This keeps the permanent tree small (no WEBFURL bloat)
            if let Some(ctx) = self.current_context_node.take() {
                ctx.detach();
            }
            let clean_assistant = user_anchor.add_assistant(response_text.as_str());

            if let Some(extra) = cmd_results {
                println!("\n--- Agent action result ---\n{extra}");

                // If a CAPTCHA was detected, stop the follow-up loop.
                // The user needs to solve it manually; don't let the LLM retry.
                if self.pending_captcha {
                    info!(depth, "CAPTCHA detected — stopping follow-up loop");
                    self.conversation_tip = clean_assistant.clone();
                    return Ok(clean_assistant);
                }

                if any_success {
                    info!(depth, "follow-up turn with command results");
                    self.completion_loop(
                        &clean_assistant,
                        Some(format!("[Command results]\n{extra}")),
                        depth + 1,
                    ).await
                } else {
                    info!(depth, "action failed — letting model retry");
                    self.completion_loop(
                        &clean_assistant,
                        Some(format!("[Error] Your action `{response_text}` failed:\n{extra}\nDo NOT repeat the same action. Try a different approach.")),
                        depth + 1,
                    ).await
                }
            } else {
                // No JSON action found — plain text answer to user
                self.conversation_tip = clean_assistant.clone();
                Ok(clean_assistant)
            }
        })
    }

    /// Stream LLM response to terminal. Updates context file live.
    /// Detects JSON action objects and force-cuts the stream when one is found.
    async fn stream_to_terminal(
        &mut self,
        stream: &mut minillmlib::provider::StreamingCompletion,
    ) -> String {
        let mut response_text = String::new();
        let mut last_dump = std::time::Instant::now();
        let mut brace_depth: i32 = 0;
        let mut in_json = false;
        let mut in_string = false;
        let mut escape_next = false;

        while let Some(result) = stream.next_chunk().await {
            match result {
                Ok(chunk) => {
                    if chunk.delta.is_empty() { continue; }

                    for ch in chunk.delta.chars() {
                        response_text.push(ch);
                        print!("{ch}");

                        // Track JSON object boundaries
                        if escape_next {
                            escape_next = false;
                            continue;
                        }
                        if in_string {
                            if ch == '\\' { escape_next = true; }
                            else if ch == '"' { in_string = false; }
                            continue;
                        }
                        match ch {
                            '{' => { brace_depth += 1; in_json = true; }
                            '}' if in_json => {
                                brace_depth -= 1;
                                if brace_depth == 0 {
                                    // Complete JSON object found — force-cut
                                    let _ = io::stdout().flush();
                                    println!();
                                    self.dump_context_file_ex(Some(&response_text));
                                    return response_text;
                                }
                            }
                            '"' if in_json => { in_string = true; }
                            _ => {}
                        }
                    }

                    let _ = io::stdout().flush();

                    // Update context file periodically (every 200ms)
                    if last_dump.elapsed() >= std::time::Duration::from_millis(200) {
                        self.dump_context_file_ex(Some(&response_text));
                        last_dump = std::time::Instant::now();
                    }
                }
                Err(e) => {
                    eprintln!("\nStream error: {e}");
                    break;
                }
            }
        }

        // Final dump with complete response
        self.dump_context_file_ex(Some(&response_text));

        response_text
    }

    /// Extract a JSON action from the response and execute it.
    /// Returns (result_text, success) or (None, false) if no JSON action found.
    async fn process_agent_commands(&mut self, response: &str) -> (Option<String>, bool) {
        // Find JSON object in response
        let json_str = match (response.find('{'), response.rfind('}')) {
            (Some(start), Some(end)) if end > start => &response[start..=end],
            _ => return (None, false),
        };

        let parsed: serde_json::Value = match serde_json::from_str(json_str) {
            Ok(v) => v,
            Err(_) => return (None, false),
        };

        let action = match parsed.get("action").and_then(|v| v.as_str()) {
            Some(a) => a,
            None => return (None, false),
        };

        let result = match action {
            "navigate" => {
                let url = parsed.get("url").and_then(|v| v.as_str()).unwrap_or("");
                self.navigate(url).await
            }
            "click" => {
                let node_id = parsed.get("node_id").and_then(|v| v.as_str()).unwrap_or("");
                match self.resolve_click_selector(node_id) {
                    Ok(selector) => {
                        match self.execute_click(&selector).await {
                            Ok(msg) => Ok(msg),
                            Err(_e1) => {
                                // Fallback 1: try dom_selector (SPA may have mutated action selector)
                                info!(node_id, "action selector failed, trying dom_selector fallback");
                                let dom_result = if let Some(dom_sel) = self.current_tree.as_ref()
                                    .and_then(|t| t.find_node(node_id))
                                    .map(|n| n.dom_selector.clone())
                                {
                                    self.execute_click(&dom_sel).await
                                } else {
                                    Err("no dom_selector".into())
                                };
                                match dom_result {
                                    Ok(msg) => Ok(msg),
                                    Err(_e2) => {
                                        // Fallback 2: use semantic description + JS to find element
                                        info!(node_id, "dom_selector also failed, trying description-based click");
                                        let node_info = self.current_tree.as_ref()
                                            .and_then(|t| t.find_node(node_id));
                                        let desc = node_info.map(|n| n.summary.clone()).unwrap_or_default();
                                        // Infer role hint from action description or node context
                                        let action_desc = node_info
                                            .and_then(|n| n.actions.first())
                                            .map(|a| match a {
                                                webfurl_core::actions::Action::Click { description, .. } => description.as_str(),
                                                webfurl_core::actions::Action::Fill { description, .. } => description.as_str(),
                                                webfurl_core::actions::Action::Select { description, .. } => description.as_str(),
                                                webfurl_core::actions::Action::Toggle { description, .. } => description.as_str(),
                                            });
                                        let role_hint = if node_id.starts_with("modal") || desc.to_lowercase().contains("dialog") || desc.to_lowercase().contains("modal") {
                                            Some("dialog")
                                        } else {
                                            None
                                        };
                                        match self.browser.click_by_description(
                                            action_desc.unwrap_or(&desc),
                                            role_hint,
                                        ).await {
                                            Ok(()) => {
                                                tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                                                let _ = self.refresh_page().await;
                                                Ok(format!("Clicked #{node_id} via description match. Page re-read."))
                                            }
                                            Err(e) => Err(format!("All click strategies failed for #{node_id}: {e}"))
                                        }
                                    }
                                }
                            }
                        }
                    }
                    Err(e) => Err(e),
                }
            }
            "fill" => {
                let node_id = parsed.get("node_id").and_then(|v| v.as_str()).unwrap_or("");
                let text = parsed.get("text").and_then(|v| v.as_str()).unwrap_or("");
                match self.resolve_fill_selector(node_id) {
                    Ok(selector) => self.execute_fill(&selector, text).await,
                    Err(e) => Err(e),
                }
            }
            "unfold" => {
                let node_id = parsed.get("node_id").and_then(|v| v.as_str()).unwrap_or("");
                self.manual_unfold(node_id)
            }
            "web_search" => {
                let query = parsed.get("query").and_then(|v| v.as_str()).unwrap_or("");
                let encoded: String = query.chars().map(|c| {
                    if c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.' || c == '~' {
                        c.to_string()
                    } else if c == ' ' {
                        "+".to_string()
                    } else {
                        format!("%{:02X}", c as u32)
                    }
                }).collect();
                let url = format!("https://www.google.com/search?q={encoded}");
                self.navigate(&url).await
            }
            "page_search" | "search" => {
                let query = parsed.get("query").and_then(|v| v.as_str()).unwrap_or("");
                self.manual_search(query).await
            }
            "describe" => {
                let node_id = parsed.get("node_id").and_then(|v| v.as_str()).unwrap_or("");
                self.describe_images(node_id).await
            }
            "ask_image" => {
                let node_id = parsed.get("node_id").and_then(|v| v.as_str()).unwrap_or("");
                let question = parsed.get("question").and_then(|v| v.as_str()).unwrap_or("");
                self.ask_image(node_id, question).await
            }
            _ => Err(format!("Unknown action: {action}")),
        };

        match result {
            Ok(msg) => (Some(msg), true),
            Err(e) => (Some(e), false),
        }
    }
}

