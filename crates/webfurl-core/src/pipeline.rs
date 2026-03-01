use std::sync::atomic::{AtomicU32, Ordering};

use dom_query::Document;
use minillmlib::{ChatNode, GeneratorInfo, CompletionParameters, NodeCompletionParameters};
use serde::Deserialize;
use tracing::{info, warn, error};

use crate::actions::Action;
use crate::cache::CacheStore;
use crate::embeddings::EmbeddingClient;
use crate::error::WebfurlError;
use crate::hasher;
use crate::tree::{ImageRef, SemanticNode, SemanticTree};

/// Internal atomic counters for parallel stat collection.
struct AtomicStats {
    cache_hits: AtomicU32,
    llm_calls: AtomicU32,
}

impl AtomicStats {
    fn new() -> Self {
        Self {
            cache_hits: AtomicU32::new(0),
            llm_calls: AtomicU32::new(0),
        }
    }

    fn record_cache_hit(&self) {
        self.cache_hits.fetch_add(1, Ordering::Relaxed);
    }

    fn record_llm_call(&self) {
        self.llm_calls.fetch_add(1, Ordering::Relaxed);
    }

    fn snapshot(&self) -> (u32, u32) {
        (self.cache_hits.load(Ordering::Relaxed), self.llm_calls.load(Ordering::Relaxed))
    }
}

/// Stats from a single pipeline compression run.
#[derive(Debug, Clone)]
pub struct CompressionRunStats {
    pub raw_html_bytes: usize,
    pub clean_html_bytes: usize,
    pub total_chunks: usize,
    pub chunks_cached: u32,
    pub chunks_llm_compressed: u32,
    pub compressed_tokens: u32,
    pub full_tree_tokens: u32,
    pub page_cache_hit: bool,
    pub duration_ms: u64,
}

impl CompressionRunStats {
    pub fn compression_ratio(&self) -> f64 {
        if self.full_tree_tokens == 0 { return 0.0; }
        self.compressed_tokens as f64 / self.full_tree_tokens as f64
    }

    /// Rough estimate: 1 token ≈ 4 chars of HTML
    pub fn estimated_raw_tokens(&self) -> usize {
        self.clean_html_bytes / 4
    }

    pub fn tokens_saved_vs_raw(&self) -> i64 {
        self.estimated_raw_tokens() as i64 - self.compressed_tokens as i64
    }
}

const NOISE_TAGS: &[&str] = &[
    "script", "style", "noscript", "meta", "link",
    "head", "iframe", "canvas", "video", "audio",
    "template", "slot",
    // SVG and all its internals
    "svg", "path", "rect", "g", "circle", "ellipse", "line",
    "polyline", "polygon", "use", "defs", "clippath", "mask",
    "pattern", "symbol", "marker", "lineargradient", "radialgradient",
];

const NOISE_CLASSES: &[&str] = &[
    "hidden", "sr-only", "visually-hidden", "d-none", "invisible",
    "screenreader", "screen-reader", "offscreen",
];

/// Tags that form natural chunk boundaries in the DOM.
const SECTION_TAGS: &[&str] = &[
    "header", "footer", "nav", "main", "aside", "section", "article",
    "dialog", "form",
];

/// Tags whose direct children are repeated items (grid/list patterns).
const LIST_TAGS: &[&str] = &["ul", "ol", "tbody", "dl"];

/// Minimum HTML length for a chunk to be worth its own LLM call.
/// Below this, it gets inlined into the parent.
const MIN_CHUNK_HTML_LEN: usize = 100;

/// Maximum HTML length before we force-split a chunk into its children.
const MAX_CHUNK_HTML_LEN: usize = 4000;

/// Attributes to preserve on elements (everything else is stripped).
const KEEP_ATTRS: &[&str] = &[
    "href", "src", "alt", "title", "type", "name", "value",
    "placeholder", "id", "role", "aria-label", "for", "action",
    "method", "target", "rel", "selected", "checked", "disabled",
];

pub struct PipelineConfig {
    pub generator: GeneratorInfo,
    pub embedding_client: EmbeddingClient,
    pub max_depth: usize,
    pub min_content_length: usize,
}

// ─── DOM Chunk: intermediate tree structure for parallel processing ───

/// A chunk of DOM that will be sent to a single LLM call.
/// Chunks form a tree: leaf chunks contain raw HTML, parent chunks
/// contain their own HTML context plus child summaries.
/// An interactive DOM element found inside a chunk, with a pre-computed CSS selector.
#[derive(Debug, Clone)]
pub struct InteractiveElement {
    /// Reliable CSS selector computed from the real DOM (not LLM-generated)
    pub selector: String,
    /// Tag name (a, button, input, select, textarea)
    pub tag: String,
    /// Key attributes for context (href, aria-label, type, name, placeholder, etc.)
    pub attrs: Vec<(String, String)>,
    /// Action type inferred from the tag
    pub action_type: &'static str,
}

#[derive(Debug, Clone)]
pub struct DomChunk {
    /// CSS selector path for this chunk (e.g. "body > header")
    pub selector: String,
    /// Tag name
    pub tag: String,
    /// The outer HTML of this element (without deep children that became sub-chunks)
    pub html: String,
    /// Structural hash of this chunk's DOM subtree
    pub structural_hash: String,
    /// Child chunks (deeper sections that get their own LLM calls)
    pub children: Vec<DomChunk>,
    /// Images found directly in this chunk
    pub images: Vec<ImageRef>,
    /// Depth in the chunk tree (0 = body direct children)
    #[allow(dead_code)]
    pub depth: usize,
    /// Interactive elements found in this chunk's HTML, with pre-computed selectors
    pub interactive_elements: Vec<InteractiveElement>,
}


// ─── Main entry points ───

/// The main entry point: HTML string → SemanticTree + compression stats.
/// Uses bottom-up parallel compression: chunks the DOM, compresses leaves
/// in parallel, then assembles parents layer by layer.
pub async fn html_to_semantic_tree(
    html: &str,
    url: &str,
    config: &PipelineConfig,
) -> crate::Result<(SemanticTree, CompressionRunStats)> {
    html_to_semantic_tree_inner(html, url, config, None).await
}

/// Cache-aware entry point. No page-level cache — caching happens at the chunk
/// level inside the compression pipeline. Each chunk is keyed by sha256(html),
/// so identical HTML fragments hit cache regardless of which page they appear on.
pub async fn html_to_semantic_tree_cached(
    html: &str,
    url: &str,
    config: &PipelineConfig,
    cache: &CacheStore,
) -> crate::Result<(SemanticTree, CompressionRunStats)> {
    html_to_semantic_tree_inner(html, url, config, Some(cache)).await
}

/// Inner implementation shared by cached and non-cached paths.
async fn html_to_semantic_tree_inner(
    html: &str,
    url: &str,
    config: &PipelineConfig,
    cache: Option<&CacheStore>,
) -> crate::Result<(SemanticTree, CompressionRunStats)> {
    let start = std::time::Instant::now();
    let raw_html_bytes = html.len();
    let domain = extract_domain(url);
    info!(url, domain = %domain, "starting parallel pipeline");

    // Phase 0: Extract interactive elements from RAW HTML (before cleaning strips classes),
    // then clean DOM and chunk it.
    let (chunks, structural_hash, title, page_images, clean_html_bytes, all_interactive) = {
        // Extract interactive elements from raw HTML — has classes, data-testid, etc.
        let all_interactive = extract_interactive_elements(html, "");
        info!(count = all_interactive.len(), "extracted interactive elements from raw HTML");

        let cleaned = clean_html(html);
        let clean_len = cleaned.len();
        let doc = Document::from(cleaned.as_str());
        let structural_hash = compute_dom_structural_hash(&doc);
        let title = {
            let orig_doc = Document::from(html);
            orig_doc
                .select("title")
                .iter()
                .next()
                .map(|t| t.text().to_string())
                .unwrap_or_default()
        };
        let images = extract_images(&doc, url);
        let chunks = chunk_dom(&doc, url);
        (chunks, structural_hash, title, images, clean_len, all_interactive)
    };

    let total_chunks: usize = count_chunks(&chunks);
    let atomic_stats = AtomicStats::new();
    info!(total_chunks, "DOM chunked into {} pieces", total_chunks);

    // Phase 1-2: Bottom-up parallel compression
    let mut nodes = compress_chunks_bottom_up(chunks, url, config, cache, &atomic_stats, &all_interactive).await?;

    let (cached, llm_compressed) = atomic_stats.snapshot();
    info!(cached, llm_compressed, total_chunks, "compression complete");

    // Phase 3: Attach images
    attach_images_to_nodes(&mut nodes, page_images);

    // Phase 4: Compute embeddings in batch
    let all_summaries: Vec<String> = nodes
        .iter()
        .flat_map(|n| collect_summaries(n))
        .collect();

    if !all_summaries.is_empty() {
        let embeddings = config.embedding_client.embed_batch(&all_summaries).await?;
        let mut emb_iter = embeddings.into_iter();
        for node in &mut nodes {
            assign_embeddings(node, &mut emb_iter);
        }
    }

    let compressed_token_count: u32 = nodes.iter().map(|n| n.token_count).sum();
    let full_token_count: u32 = nodes.iter().map(|n| n.subtree_token_count).sum();

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();

    let tree = SemanticTree {
        url: url.to_string(),
        domain,
        title,
        root_nodes: nodes,
        compressed_token_count,
        full_token_count,
        structural_hash,
        created_at: now,
        dynamic_slots_filled_at: now,
    };

    let run_stats = CompressionRunStats {
        raw_html_bytes,
        clean_html_bytes,
        total_chunks,
        chunks_cached: cached,
        chunks_llm_compressed: llm_compressed,
        compressed_tokens: compressed_token_count,
        full_tree_tokens: full_token_count,
        page_cache_hit: false,
        duration_ms: start.elapsed().as_millis() as u64,
    };

    Ok((tree, run_stats))
}

// ─── Phase 0: DOM Chunking ───

/// Parse the cleaned DOM into a tree of chunks using the Node API.
/// Uses element_children() to skip text nodes and avoid nth-child index issues.
pub fn chunk_dom(doc: &Document, page_url: &str) -> Vec<DomChunk> {
    let body = doc.select("body");
    let body_node = match body.nodes().first() {
        Some(n) => n.clone(),
        None => return vec![],
    };

    let mut chunks = vec![];
    let mut tag_counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();

    for child in body_node.element_children() {
        let tag = child.node_name()
            .map(|s| s.to_string().to_lowercase())
            .unwrap_or_else(|| "div".to_string());

        let count = tag_counts.entry(tag.clone()).or_insert(0);
        *count += 1;
        let nth = *count;

        // Build a CSS selector: body > tag:nth-of-type(n)
        let selector = if nth == 1 {
            format!("body > {tag}")
        } else {
            format!("body > {tag}:nth-of-type({nth})")
        };

        if let Some(chunk) = chunk_node(&child, &selector, &tag, page_url, 0) {
            chunks.push(chunk);
        }
    }

    chunks
}

/// Recursively chunk a single DOM node.
/// Uses the Node API directly to avoid CSS selector issues.
fn chunk_node(
    node: &dom_query::Node,
    selector: &str,
    tag: &str,
    page_url: &str,
    depth: usize,
) -> Option<DomChunk> {
    // Collapse wrapper divs: if this is a div/span with exactly one element child
    // and no direct text, skip this wrapper and chunk the child directly.
    // Depth is NOT incremented here — collapsing wrappers is free, not real nesting.
    if tag == "div" || tag == "span" {
        let children = node.element_children();
        let direct_text: String = node.children()
            .into_iter()
            .filter(|c| c.is_text())
            .map(|c| c.text().to_string())
            .collect::<Vec<_>>()
            .join("");
        if children.len() == 1 && direct_text.trim().is_empty() {
            let child = &children[0];
            let child_tag = child.node_name()
                .map(|s| s.to_string().to_lowercase())
                .unwrap_or_else(|| "div".to_string());
            let child_selector = format!("{selector} > {child_tag}");
            return chunk_node(child, &child_selector, &child_tag, page_url, depth);
        }
    }

    let outer_html = node.html().to_string();

    // Skip tiny noise elements
    let text_content = node.text().to_string();
    if text_content.trim().is_empty() {
        // Check if it has any interactive or media elements
        let inner = node.inner_html().to_string();
        let has_interactive = inner.contains("<img")
            || inner.contains("<input")
            || inner.contains("<button")
            || inner.contains("<a ");
        if !has_interactive {
            return None;
        }
    }

    // Extract images from this node's HTML
    let images = {
        let frag = Document::from(outer_html.as_str());
        extract_images(&frag, page_url)
    };

    // Compute structural hash for this element
    let structural_hash = {
        let frag = Document::from(outer_html.as_str());
        compute_dom_structural_hash(&frag)
    };

    let element_children = node.element_children();

    // Decide whether to split into sub-chunks
    let should_split = outer_html.len() > MAX_CHUNK_HTML_LEN
        || SECTION_TAGS.contains(&tag)
        || LIST_TAGS.contains(&tag)
        || is_grid_node(node);

    if outer_html.len() > MAX_CHUNK_HTML_LEN {
        info!(
            selector = %selector, tag = %tag, depth,
            html_len = outer_html.len(),
            n_children = element_children.len(),
            should_split,
            "large node encountered"
        );
    }

    if should_split && depth < 8 && !element_children.is_empty() {
        let mut child_chunks = vec![];
        let mut inlined_children_html = String::new();
        let mut child_tag_counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();

        for child in &element_children {
            let child_tag = child.node_name()
                .map(|s| s.to_string().to_lowercase())
                .unwrap_or_else(|| "div".to_string());

            let count = child_tag_counts.entry(child_tag.clone()).or_insert(0);
            *count += 1;
            let nth = *count;

            let child_selector = if nth == 1 {
                format!("{selector} > {child_tag}")
            } else {
                format!("{selector} > {child_tag}:nth-of-type({nth})")
            };

            let child_html = child.html().to_string();

            if child_html.len() >= MIN_CHUNK_HTML_LEN {
                if let Some(child_chunk) = chunk_node(child, &child_selector, &child_tag, page_url, depth + 1) {
                    child_chunks.push(child_chunk);
                }
            } else {
                inlined_children_html.push_str(&child_html);
            }
        }

        if !child_chunks.is_empty() {
            let shallow_html = build_shallow_html_from_node(tag, node, &inlined_children_html);

            return Some(DomChunk {
                selector: selector.to_string(),
                tag: tag.to_string(),
                html: shallow_html,
                structural_hash,
                children: child_chunks,
                images,
                depth,
                interactive_elements: vec![],
            });
        }
    }

    // Leaf chunk
    Some(DomChunk {
        selector: selector.to_string(),
        tag: tag.to_string(),
        html: outer_html,
        structural_hash,
        children: vec![],
        images,
        depth,
        interactive_elements: vec![],
    })
}

/// Build a shallow HTML string for a parent chunk from a Node.
fn build_shallow_html_from_node(
    tag: &str,
    node: &dom_query::Node,
    inlined_children_html: &str,
) -> String {
    let mut attr_str = String::new();
    for attr in node.attrs() {
        attr_str.push_str(&format!(
            " {}=\"{}\"",
            attr.name.local,
            attr.value
        ));
    }

    // Get direct text content (text node children only, not nested elements)
    let direct_text: String = node.children()
        .into_iter()
        .filter(|c| c.is_text())
        .map(|c| c.text().to_string())
        .collect::<Vec<_>>()
        .join(" ");

    let direct_text_trimmed = direct_text.trim();

    format!(
        "<{tag}{attr_str}>{}{}</{tag}>",
        if direct_text_trimmed.is_empty() { "" } else { direct_text_trimmed },
        inlined_children_html,
    )
}

/// Check if a node looks like a grid/list container (many similar direct children).
fn is_grid_node(node: &dom_query::Node) -> bool {
    let children = node.element_children();
    let count = children.len();
    if count < 4 {
        return false;
    }

    let mut tags: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
    for child in &children {
        let tag = child.node_name()
            .map(|s| s.to_string().to_lowercase())
            .unwrap_or_default();
        *tags.entry(tag).or_insert(0) += 1;
    }

    tags.values().any(|&v| v as f32 / count as f32 > 0.6)
}

/// Interactive tag names we look for in leaf chunks.
const INTERACTIVE_TAGS: &[&str] = &["a", "button", "input", "select", "textarea"];

/// Attributes worth keeping on interactive elements for LLM context.
const INTERACTIVE_ATTRS: &[&str] = &[
    "href", "aria-label", "title", "type", "name", "placeholder",
    "role", "value", "id", "for", "action", "method",
];


/// Extract interactive elements from a chunk's HTML.
/// Returns a list of `InteractiveElement` with SHORT, attribute-based CSS selectors
/// that work against the live browser DOM.
///
/// IMPORTANT: These selectors have NO structural parent path (no "body > div > ...").
/// They use only the element's own attributes: id, href, aria-label, data-testid, class.
/// This makes them resilient to React re-renders and SPA DOM mutations.
///
/// `raw_html` should be the RAW browser HTML (before clean_html stripping), so we
/// have access to classes, data-testid, etc.
fn extract_interactive_elements(raw_html: &str, _chunk_selector: &str) -> Vec<InteractiveElement> {
    let doc = Document::from(raw_html);
    let mut results = vec![];

    for &itag in INTERACTIVE_TAGS {
        let selection = doc.select(itag);
        let nodes: Vec<_> = selection.nodes().to_vec();
        if nodes.is_empty() {
            continue;
        }

        for node in nodes.iter() {
            // Collect attributes for LLM context
            let mut attrs = vec![];
            for &attr_name in INTERACTIVE_ATTRS {
                if let Some(val) = node.attr(attr_name) {
                    let val_str = val.to_string();
                    if !val_str.is_empty() {
                        attrs.push((attr_name.to_string(), val_str));
                    }
                }
            }

            // Build a SHORT, attribute-only selector. No parent chain.
            // Priority: id > href (for <a>) > aria-label > data-testid > name > class combo
            let selector = build_short_selector(itag, &node);

            // Skip elements where we couldn't build any meaningful selector
            if selector.is_none() {
                continue;
            }
            let selector = selector.unwrap();

            let action_type = match itag {
                "input" | "textarea" => {
                    let input_type = node.attr("type")
                        .map(|v| v.to_string().to_lowercase())
                        .unwrap_or_else(|| "text".to_string());
                    match input_type.as_str() {
                        "checkbox" | "radio" => "Toggle",
                        "submit" | "button" | "reset" | "image" => "Click",
                        _ => "Fill",
                    }
                }
                "select" => "Select",
                _ => "Click", // a, button
            };

            results.push(InteractiveElement {
                selector,
                tag: itag.to_string(),
                attrs,
                action_type,
            });
        }
    }

    results
}

/// Build a short, attribute-based CSS selector for an interactive element.
/// Uses only the element's own attributes — no structural path.
/// Returns None if we can't build anything meaningful.
fn build_short_selector(tag: &str, node: &dom_query::Node) -> Option<String> {
    // 1. id — globally unique
    if let Some(id) = node.attr("id") {
        let id_str = id.to_string();
        if !id_str.is_empty() && !id_str.contains(' ') {
            return Some(format!("#{id_str}"));
        }
    }

    // 2. For <a>: href path (strip query params, use contains match)
    if tag == "a" {
        if let Some(href) = node.attr("href") {
            let href_str = href.to_string();
            if !href_str.is_empty() && href_str != "#" && href_str != "/" && href_str != "javascript:void(0)" {
                let path = href_str.split('?').next().unwrap_or(&href_str);
                if path.len() > 1 {
                    let escaped = path.replace('\'', "\\'");
                    return Some(format!("a[href*='{escaped}']"));
                }
            }
        }
    }

    // 3. aria-label — usually unique per element type
    if let Some(aria) = node.attr("aria-label") {
        let aria_str = aria.to_string();
        if !aria_str.is_empty() {
            let escaped = aria_str.replace('\'', "\\'");
            return Some(format!("{tag}[aria-label='{escaped}']"));
        }
    }

    // 4. data-testid — stable test attribute, survives re-renders
    if let Some(testid) = node.attr("data-testid") {
        let testid_str = testid.to_string();
        if !testid_str.is_empty() {
            return Some(format!("[data-testid='{testid_str}']"));
        }
    }

    // 5. name attribute (for inputs/forms)
    if let Some(name) = node.attr("name") {
        let name_str = name.to_string();
        if !name_str.is_empty() {
            return Some(format!("{tag}[name='{name_str}']"));
        }
    }

    // 6. For <input>: type + placeholder combo
    if tag == "input" || tag == "textarea" {
        if let Some(placeholder) = node.attr("placeholder") {
            let ph_str = placeholder.to_string();
            if !ph_str.is_empty() {
                let escaped = ph_str.replace('\'', "\\'");
                let type_str = node.attr("type").map(|v| v.to_string()).unwrap_or_default();
                if !type_str.is_empty() {
                    return Some(format!("{tag}[type='{type_str}'][placeholder*='{escaped}']"));
                } else {
                    return Some(format!("{tag}[placeholder*='{escaped}']"));
                }
            }
        }
    }

    // 7. role + class combination as last resort
    if let Some(role) = node.attr("role") {
        let role_str = role.to_string();
        if !role_str.is_empty() && role_str != "presentation" {
            if let Some(class) = node.attr("class") {
                let class_str = class.to_string();
                // Pick the first non-generic class
                if let Some(cls) = class_str.split_whitespace()
                    .find(|c| c.len() > 3 && !["div", "span", "btn"].contains(c))
                {
                    return Some(format!("{tag}[role='{role_str}'].{cls}"));
                }
            }
            return Some(format!("{tag}[role='{role_str}']"));
        }
    }

    // 8. class-only as absolute last resort (pick the most specific class)
    if let Some(class) = node.attr("class") {
        let class_str = class.to_string();
        if let Some(cls) = class_str.split_whitespace()
            .filter(|c| c.len() > 4)
            .max_by_key(|c| c.len())
        {
            return Some(format!("{tag}.{cls}"));
        }
    }

    // Can't build a useful selector — skip this element
    None
}

/// Check if an interactive element belongs to a chunk by matching its identifying
/// attributes against the chunk's cleaned HTML content.
///
/// For hrefs: we extract a "discriminating" portion (e.g. room ID from "/rooms/123456")
/// that's specific enough to avoid cross-chunk false positives.
/// For aria-labels: we require the full label text appears in the chunk.
fn element_matches_chunk(el: &InteractiveElement, chunk_html: &str) -> bool {
    for (k, v) in &el.attrs {
        match k.as_str() {
            "href" => {
                // Extract the most specific path segment as discriminator.
                // e.g. "/rooms/1156986992162714739" → "1156986992162714739"
                // e.g. "/s/Mountain-View--CA/homes" → too generic, skip
                let path = v.split('?').next().unwrap_or(v);
                if let Some(discriminator) = extract_href_discriminator(path) {
                    if chunk_html.contains(&discriminator) {
                        return true;
                    }
                }
            }
            "aria-label" => {
                // Require the full aria-label to appear in the chunk HTML.
                // This prevents "Add to wishlist: Room in Mountain View" from
                // matching unrelated chunks that happen to contain "Mountain View".
                if !v.is_empty() && v.len() > 3 && chunk_html.contains(v.as_str()) {
                    return true;
                }
            }
            "title" | "placeholder" | "name" => {
                if !v.is_empty() && v.len() > 3 && chunk_html.contains(v.as_str()) {
                    return true;
                }
            }
            _ => {}
        }
    }
    // Try matching by id
    if let Some((_, id_val)) = el.attrs.iter().find(|(k, _)| k == "id") {
        if !id_val.is_empty() && chunk_html.contains(id_val.as_str()) {
            return true;
        }
    }
    false
}

/// Extract a discriminating portion from an href path.
/// Returns a string that's specific enough to uniquely identify the link
/// without matching unrelated chunks.
///
/// Strategy: find the last path segment that looks like an ID (numeric or long alphanumeric).
/// Falls back to the full path if it's short enough to be specific.
fn extract_href_discriminator(path: &str) -> Option<String> {
    let segments: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();
    if segments.is_empty() {
        return None;
    }

    // Look for numeric-heavy segments (IDs like "1156986992162714739")
    for &seg in segments.iter().rev() {
        if seg.len() >= 6 && seg.chars().filter(|c| c.is_ascii_digit()).count() > seg.len() / 2 {
            return Some(seg.to_string());
        }
    }

    // If path has 3+ segments and the last is specific, use it
    if segments.len() >= 2 {
        let last = segments.last().unwrap();
        if last.len() >= 5 {
            return Some(last.to_string());
        }
    }

    // Short paths like "/login" or "/help" — use the full path
    if path.len() > 1 && path.len() < 30 {
        return Some(path.to_string());
    }

    None
}

fn resolve_url(src: &str, page_url: &str, base_url: &str) -> String {
    if src.starts_with("http://") || src.starts_with("https://") {
        src.to_string()
    } else if src.starts_with("//") {
        format!("https:{src}")
    } else if src.starts_with('/') {
        let host = page_url
            .find("://")
            .and_then(|i| page_url[i + 3..].find('/').map(|j| &page_url[..i + 3 + j]))
            .unwrap_or(base_url);
        format!("{host}{src}")
    } else {
        format!("{base_url}/{src}")
    }
}

pub fn count_chunks(chunks: &[DomChunk]) -> usize {
    chunks.iter().map(|c| 1 + count_chunks(&c.children)).sum()
}

// ─── Phase 1-2: Bottom-up parallel compression ───

/// Compress all chunks bottom-up: leaves first (in parallel), then parents.
async fn compress_chunks_bottom_up(
    chunks: Vec<DomChunk>,
    url: &str,
    config: &PipelineConfig,
    cache: Option<&CacheStore>,
    stats: &AtomicStats,
    all_interactive: &[InteractiveElement],
) -> crate::Result<Vec<SemanticNode>> {
    // Launch all root-level compressions in parallel
    let futures: Vec<_> = chunks.into_iter().enumerate().map(|(i, chunk)| {
        compress_chunk_recursive(chunk, url, "", i, config, cache, stats, all_interactive)
    }).collect();

    let all_results = futures::future::join_all(futures).await;

    let mut results: Vec<(usize, crate::Result<SemanticNode>)> = all_results
        .into_iter()
        .enumerate()
        .collect();

    // Sort by original order and collect
    results.sort_by_key(|(i, _)| *i);
    let mut nodes = vec![];
    let mut failed = 0u32;
    for (i, result) in results {
        match result {
            Ok(node) => nodes.push(node),
            Err(e) => {
                failed += 1;
                error!(chunk_index = i, error = %e, "chunk compression FAILED — node dropped");
            }
        }
    }
    if failed > 0 {
        error!(failed, total = nodes.len() + failed as usize, "⚠️  {failed} chunks failed compression and were dropped");
    }

    Ok(nodes)
}

/// Recursively compress a chunk: compress children first (in parallel), then self.
fn compress_chunk_recursive<'a>(
    chunk: DomChunk,
    url: &'a str,
    id_prefix: &'a str,
    index: usize,
    config: &'a PipelineConfig,
    cache: Option<&'a CacheStore>,
    stats: &'a AtomicStats,
    all_interactive: &'a [InteractiveElement],
) -> std::pin::Pin<Box<dyn std::future::Future<Output = crate::Result<SemanticNode>> + Send + 'a>> {
    Box::pin(async move {
        // Content hash = sha256(chunk.html) — the cache key
        let content_key = hasher::content_hash(&chunk.html);

        // If this chunk has no children, it's a leaf → compress directly
        if chunk.children.is_empty() {
            return compress_leaf_chunk(&chunk, url, id_prefix, index, config, cache, stats, &content_key, all_interactive).await;
        }

        // Always compress children first (bottom-up) — children are never served from parent cache
        let child_futures: Vec<_> = chunk.children.into_iter().enumerate().map(|(ci, child)| {
            compress_chunk_recursive(child, url, id_prefix, ci, config, cache, stats, all_interactive)
        }).collect();

        let child_results = futures::future::join_all(child_futures).await;

        let mut child_nodes = vec![];
        for result in child_results {
            match result {
                Ok(node) => child_nodes.push(node),
                Err(e) => error!(selector = %chunk.selector, error = %e, "child chunk compression FAILED"),
            }
        }

        // Check parent cache: if this parent's own HTML hasn't changed, reuse its summary
        // (children are always fresh from above — we only cache the parent's own metadata)
        if let Some(cache) = cache {
            if let Ok(Some(mut cached_parent)) = cache.get_chunk(&content_key).await {
                info!(selector = %chunk.selector, "� parent cache hit — reusing summary, attaching fresh children");
                stats.record_cache_hit();

                // Re-apply computed fields (ID from selector, not from cache)
                cached_parent.id = selector_to_id(&chunk.selector);
                cached_parent.dom_selector = chunk.selector.clone();
                if cached_parent.is_dynamic {
                    cached_parent.dynamic_selector = Some(chunk.selector.clone());
                }

                // Attach freshly-compressed children
                cached_parent.children = child_nodes;
                cached_parent.subtree_token_count = cached_parent.token_count
                    + cached_parent.children.iter().map(|c| c.subtree_token_count).sum::<u32>();

                return Ok(cached_parent);
            }
        }

        // Cache miss — compress parent with LLM
        stats.record_llm_call();
        let parent_node = compress_parent_chunk(
            &chunk.html,
            &chunk.selector,
            &chunk.tag,
            &chunk.structural_hash,
            &chunk.images,
            &child_nodes,
            url,
            id_prefix,
            index,
            config,
        ).await?;

        // Cache the parent (without children — they're always recomputed)
        if let Some(cache) = cache {
            let _ = cache.put_chunk(&content_key, &parent_node, true).await;
        }

        Ok(parent_node)
    })
}

/// Compress a leaf chunk (no sub-chunks) via a single LLM call.
async fn compress_leaf_chunk(
    chunk: &DomChunk,
    _url: &str,
    _id_prefix: &str,
    _index: usize,
    config: &PipelineConfig,
    cache: Option<&CacheStore>,
    stats: &AtomicStats,
    content_key: &str,
    all_interactive: &[InteractiveElement],
) -> crate::Result<SemanticNode> {
    // Check chunk cache by content hash
    if let Some(cache) = cache {
        if let Ok(Some(mut cached_node)) = cache.get_chunk(content_key).await {
            info!(selector = %chunk.selector, "🟢 leaf cache hit");
            stats.record_cache_hit();
            // Re-apply computed fields (ID from selector)
            fixup_node(&mut cached_node, &chunk.selector);
            return Ok(cached_node);
        }
    }

    stats.record_llm_call();

    // Match interactive elements from the global list to this chunk.
    // An element belongs to this chunk if the chunk's HTML contains the element's
    // identifying text (href path, aria-label, name, placeholder, etc.)
    // Deduplicate by selector — multiple raw DOM elements may produce the same selector.
    let mut seen_selectors = std::collections::HashSet::new();
    let matched_elements: Vec<&InteractiveElement> = all_interactive.iter()
        .filter(|el| element_matches_chunk(el, &chunk.html))
        .filter(|el| seen_selectors.insert(el.selector.clone()))
        .collect();

    info!(
        selector = %chunk.selector,
        html_len = chunk.html.len(),
        interactive = matched_elements.len(),
        "compressing leaf chunk"
    );

    // Extract raw visible text from HTML for ground-truth storage (sync block — dom_query is !Send)
    let (raw_text, raw_text_tokens) = extract_raw_text(&chunk.html);

    // Build the interactive elements section for the prompt
    let interactive_section = if matched_elements.is_empty() {
        String::from("\nThis chunk has NO interactive elements. Output an empty \"element_descriptions\" array.")
    } else {
        let mut s = format!("\nThis chunk has {} interactive elements (extracted from the DOM):\n", matched_elements.len());
        for (i, el) in matched_elements.iter().enumerate() {
            let attr_str: String = el.attrs.iter()
                .map(|(k, v)| {
                    let v_short = if v.len() > 80 { format!("{}...", &v[..77]) } else { v.clone() };
                    format!("  {}=\"{}\"", k, v_short)
                })
                .collect::<Vec<_>>()
                .join("");
            s.push_str(&format!("  [{i}] <{tag}>{attr_str} → {action_type}\n",
                tag = el.tag, action_type = el.action_type));
        }
        s.push_str("\nFor each element, provide a short description of what it does (e.g. \"Open listing page\", \"Toggle wishlist\").");
        s
    };

    let prompt = format!(
        r#"Compress this HTML fragment into a JSON object.

This is a LEAF node (deepest level). Your summary should be DETAILED and SPECIFIC.
Include actual content: names, prices, values, labels, counts.

Output a JSON object with:
- "summary": detailed description (5-40 tokens). Include specific content (names, prices, ratings).
- "is_dynamic": true if content changes between visits
- "stable": true ONLY if summary describes a fixed structure (e.g. "search input"), false if it contains specific data
- "element_descriptions": array of strings, one per interactive element listed below, in the SAME ORDER. Each string describes what the element does (5-15 tokens).
{interactive_section}

Example output:
{{
  "summary": "Mountain View home, 2BR, $1,680/night, 4.91 rating",
  "is_dynamic": true, "stable": false,
  "element_descriptions": ["Open the listing page", "Toggle wishlist", "Show next photo"]
}}

HTML:
{html}

Output ONE JSON object. No markdown. No explanation."#,
        html = chunk.html,
        interactive_section = interactive_section,
    );

    let json = llm_call(&prompt, config).await?;

    // Parse the simplified response
    let parsed: LeafResponse = serde_json::from_str(&json)
        .map_err(|e| WebfurlError::LlmError(format!("Failed to parse leaf response: {e}\nRaw: {json}")))?;

    // Build interactive children from pre-computed elements + LLM descriptions
    let mut children = vec![];
    for (i, el) in matched_elements.iter().enumerate() {
        let description = parsed.element_descriptions
            .get(i)
            .cloned()
            .unwrap_or_else(|| format!("{} element", el.tag));

        let action = match el.action_type {
            "Fill" => Action::Fill {
                selector: el.selector.clone(),
                field_type: {
                    let input_type = el.attrs.iter()
                        .find(|(k, _)| k == "type")
                        .map(|(_, v)| v.as_str())
                        .unwrap_or("text");
                    match input_type {
                        "password" => crate::actions::FieldType::Password,
                        "email" => crate::actions::FieldType::Email,
                        "number" => crate::actions::FieldType::Number,
                        "search" => crate::actions::FieldType::Search,
                        "url" => crate::actions::FieldType::Url,
                        _ => crate::actions::FieldType::Text,
                    }
                },
                description: description.clone(),
            },
            "Select" => Action::Select {
                selector: el.selector.clone(),
                options: vec![],
                description: description.clone(),
            },
            "Toggle" => Action::Toggle {
                selector: el.selector.clone(),
                description: description.clone(),
                current_state: false,
            },
            _ => Action::Click {
                selector: el.selector.clone(),
                description: description.clone(),
            },
        };

        let child_id = format!("{}-c{i}", selector_to_id(&chunk.selector));
        let token_count = estimate_tokens(&description);
        children.push(SemanticNode {
            id: child_id,
            summary: description,
            embedding: vec![],
            structural_hash: String::new(),
            is_dynamic: false,
            dynamic_selector: None,
            children: vec![],
            actions: vec![action],
            images: vec![],
            dom_selector: el.selector.clone(),
            token_count,
            subtree_token_count: token_count,
            stable: false,
            raw_text: None,
            raw_text_tokens: 0,
        });
    }

    let parent_id = selector_to_id(&chunk.selector);
    let summary_tokens = estimate_tokens(&parsed.summary);
    let subtree_tokens = summary_tokens
        + children.iter().map(|c| c.subtree_token_count).sum::<u32>()
        + raw_text_tokens;

    let node = SemanticNode {
        id: parent_id,
        summary: parsed.summary,
        embedding: vec![],
        structural_hash: chunk.structural_hash.clone(),
        is_dynamic: parsed.is_dynamic,
        dynamic_selector: if parsed.is_dynamic { Some(chunk.selector.clone()) } else { None },
        children,
        actions: vec![],
        images: vec![],
        dom_selector: chunk.selector.clone(),
        token_count: summary_tokens,
        subtree_token_count: subtree_tokens,
        stable: parsed.stable,
        raw_text,
        raw_text_tokens,
    };

    // Cache the result (leaf = complete node with interactive children)
    if let Some(cache) = cache {
        let _ = cache.put_chunk(content_key, &node, false).await;
    }

    Ok(node)
}

/// Compress a parent chunk: its own HTML context + summaries of child nodes.
async fn compress_parent_chunk(
    _shallow_html: &str,
    selector: &str,
    _tag: &str,
    structural_hash: &str,
    _images: &[ImageRef],
    child_nodes: &[SemanticNode],
    _url: &str,
    _id_prefix: &str,
    _index: usize,
    config: &PipelineConfig,
) -> crate::Result<SemanticNode> {
    // Build child summaries for context
    let child_context: String = child_nodes.iter().enumerate().map(|(i, child)| {
        let actions: Vec<&str> = child.actions.iter().map(|a| match a {
            Action::Click { .. } => "click",
            Action::Fill { .. } => "fill",
            Action::Select { .. } => "select",
            Action::Toggle { .. } => "toggle",
        }).collect();
        let action_str = if actions.is_empty() {
            String::new()
        } else {
            format!(" ({})", actions.join(","))
        };
        format!("  Child {i}: [{id}] {summary}{action_str} ({n_children} children)",
            id = child.id,
            summary = child.summary,
            n_children = child.children.len(),
        )
    }).collect::<Vec<_>>().join("\n");

    let prompt = format!(
        r#"Compress this HTML section into a JSON semantic node.
This is a PARENT node — it contains sub-sections that are already compressed below.

Your summary should be STRUCTURAL and GENERIC: describe WHAT KIND of content is here,
not the specific content itself.
Good: "Grid of apartment listing cards with photos and prices"
Bad: "3 apartments in Paris ranging from $100-200/night" (too specific)

IDs and selectors are computed automatically. Do NOT generate them.

Output a single JSON object with:
- "summary": STRUCTURAL description (5-20 tokens)
- "is_dynamic": true if content changes between visits
- "stable": true if this layout description stays the same even when children change
- "children": [] (leave empty — children are already processed)
- "actions": [] (no actions on container nodes)

Already-compressed children:
{child_context}

Output ONE JSON object. No markdown. No explanation."#,
    );

    let json = llm_call(&prompt, config).await?;
    let mut parent = parse_single_node(&json, structural_hash)?;

    // Override computed fields: ID and dom_selector from chunk's CSS selector
    parent.id = selector_to_id(selector);
    parent.dom_selector = selector.to_string();
    if parent.is_dynamic {
        parent.dynamic_selector = Some(selector.to_string());
    }

    // Attach the already-compressed children
    parent.children = child_nodes.to_vec();

    // Recompute token counts
    parent.subtree_token_count = parent.token_count
        + parent.children.iter().map(|c| c.subtree_token_count).sum::<u32>();

    Ok(parent)
}

// ─── LLM call helper ───

async fn llm_call(prompt: &str, config: &PipelineConfig) -> crate::Result<String> {
    let system = ChatNode::root(
        "You compress HTML into structured JSON. Output valid JSON only. Never output markdown or commentary.",
    );
    let user = system.add_user(prompt);

    let params = NodeCompletionParameters::new()
        .with_parse_json(true)
        .with_retry(3)
        .with_params(
            CompletionParameters::new()
                .with_temperature(0.0)
                .with_max_tokens(4096),
        );

    let response = user
        .complete(&config.generator, Some(&params))
        .await
        .map_err(|e| WebfurlError::LlmError(e.to_string()))?;

    response
        .text()
        .ok_or_else(|| WebfurlError::LlmError("empty LLM response".to_string()))
        .map(|s| s.to_string())
}

/// Simplified leaf response: LLM only provides summary + descriptions for pre-extracted elements.
#[derive(Deserialize)]
struct LeafResponse {
    #[serde(default)]
    summary: String,
    #[serde(default)]
    is_dynamic: bool,
    #[serde(default)]
    stable: bool,
    #[serde(default)]
    element_descriptions: Vec<String>,
}

// ─── Parsing (used by parent prompt only now) ───

#[derive(Deserialize)]
struct RawNode {
    #[serde(default)]
    id: String,
    #[serde(default)]
    summary: String,
    #[serde(default)]
    is_dynamic: bool,
    #[serde(default)]
    stable: bool,
    #[serde(default)]
    dom_selector: String,
    #[serde(default)]
    children: Vec<RawNode>,
    #[serde(default)]
    actions: Vec<RawAction>,
}

#[derive(Deserialize)]
struct RawAction {
    #[serde(rename = "type")]
    action_type: String,
    #[serde(default)]
    selector: String,
    #[serde(default)]
    description: String,
    #[serde(default)]
    options: Vec<String>,
    #[serde(default)]
    field_type: Option<String>,
    #[serde(default)]
    current_state: Option<bool>,
}

/// Parse a single node JSON (possibly with children) from an LLM response.
fn parse_single_node(json: &str, structural_hash: &str) -> crate::Result<SemanticNode> {
    // Try parsing as a single object first, then as an array (take first)
    let raw: RawNode = match serde_json::from_str(json) {
        Ok(node) => node,
        Err(_) => {
            // Maybe LLM returned an array
            let nodes: Vec<RawNode> = serde_json::from_str(json)?;
            nodes.into_iter().next()
                .ok_or_else(|| WebfurlError::LlmError("empty node array".to_string()))?
        }
    };

    Ok(raw_to_semantic(raw, structural_hash))
}

fn raw_to_semantic(raw: RawNode, structural_hash: &str) -> SemanticNode {
    let children: Vec<SemanticNode> = raw.children.into_iter().map(|c| {
        let child_hash = hasher::structural_hash(&c.id, &[], &[]);
        raw_to_semantic(c, &child_hash)
    }).collect();

    let actions: Vec<Action> = raw
        .actions
        .into_iter()
        .map(|a| match a.action_type.to_lowercase().as_str() {
            "fill" => Action::Fill {
                selector: a.selector,
                field_type: match a.field_type.as_deref() {
                    Some("password") => crate::actions::FieldType::Password,
                    Some("email") => crate::actions::FieldType::Email,
                    Some("number") => crate::actions::FieldType::Number,
                    Some("search") => crate::actions::FieldType::Search,
                    Some("url") => crate::actions::FieldType::Url,
                    Some("textarea") => crate::actions::FieldType::Textarea,
                    _ => crate::actions::FieldType::Text,
                },
                description: a.description,
            },
            "select" => Action::Select {
                selector: a.selector,
                options: a.options,
                description: a.description,
            },
            "toggle" => Action::Toggle {
                selector: a.selector,
                description: a.description,
                current_state: a.current_state.unwrap_or(false),
            },
            _ => Action::Click {
                selector: a.selector,
                description: a.description,
            },
        })
        .collect();

    let token_count = estimate_tokens(&raw.summary);
    let subtree_token_count = token_count
        + children.iter().map(|c| c.subtree_token_count).sum::<u32>();

    SemanticNode {
        id: raw.id,
        summary: raw.summary,
        embedding: vec![],
        structural_hash: structural_hash.to_string(),
        is_dynamic: raw.is_dynamic,
        dynamic_selector: if raw.is_dynamic {
            Some(raw.dom_selector.clone())
        } else {
            None
        },
        children,
        actions,
        images: vec![],
        dom_selector: raw.dom_selector,
        token_count,
        subtree_token_count,
        stable: raw.stable,
        raw_text: None,
        raw_text_tokens: 0,
    }
}

// ─── Utility functions ───

/// Compute a short, deterministic ID from a CSS selector.
/// e.g. "body > div:nth-of-type(2) > main > div" → "n-2mdk" ("n-" prefix + 4-char hash)
fn selector_to_id(selector: &str) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    selector.hash(&mut h);
    let hash = h.finish();
    // Base36 encode the lower 20 bits for a short 4-char suffix
    let short = hash & 0xFFFFF;
    format!("n-{}", base36(short))
}

fn base36(mut n: u64) -> String {
    if n == 0 { return "0".to_string(); }
    const CHARS: &[u8] = b"0123456789abcdefghijklmnopqrstuvwxyz";
    let mut s = Vec::new();
    while n > 0 {
        s.push(CHARS[(n % 36) as usize]);
        n /= 36;
    }
    s.reverse();
    String::from_utf8(s).unwrap()
}

/// After LLM parsing, override computed fields on a node and its children.
/// - Sets `id` from the chunk selector
/// - Sets `dom_selector` from the chunk selector
/// - For children: sets `id` as parent_id + "-c" + index
///   and keeps the LLM's dom_selector (it's the interactive element's selector)
fn fixup_node(node: &mut SemanticNode, chunk_selector: &str) {
    let parent_id = selector_to_id(chunk_selector);
    node.id = parent_id.clone();
    node.dom_selector = chunk_selector.to_string();
    if node.is_dynamic {
        node.dynamic_selector = Some(chunk_selector.to_string());
    }
    for (i, child) in node.children.iter_mut().enumerate() {
        // Child keeps its LLM-generated dom_selector (points to the interactive element)
        // but gets a deterministic ID
        child.id = format!("{parent_id}-c{i}");
    }
}

pub fn estimate_tokens(text: &str) -> u32 {
    (text.len() as u32 / 4).max(1)
}

fn extract_raw_text(html: &str) -> (Option<String>, u32) {
    let doc = dom_query::Document::from(html);
    let text = doc.select("body").text().to_string();
    let text = if text.trim().is_empty() {
        doc.root().text().to_string()
    } else {
        text
    };
    let trimmed = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if trimmed.is_empty() {
        (None, 0)
    } else {
        let tokens = estimate_tokens(&trimmed);
        (Some(trimmed), tokens)
    }
}

fn collect_summaries(node: &SemanticNode) -> Vec<String> {
    let mut result = vec![node.summary.clone()];
    for child in &node.children {
        result.extend(collect_summaries(child));
    }
    result
}

fn assign_embeddings(
    node: &mut SemanticNode,
    emb_iter: &mut impl Iterator<Item = Vec<f32>>,
) {
    match emb_iter.next() {
        Some(emb) => node.embedding = emb,
        None => warn!(node_id = %node.id, "embedding iterator exhausted"),
    }
    for child in &mut node.children {
        assign_embeddings(child, emb_iter);
    }
}

/// Remove noise elements from the DOM and strip useless attributes.
pub fn clean_html(html: &str) -> String {
    let doc = Document::from(html);

    // Remove noise tags
    for tag in NOISE_TAGS {
        doc.select(tag).remove();
    }

    // Remove noise classes
    for class in NOISE_CLASSES {
        let selector = format!(".{class}");
        doc.select(&selector).remove();
    }

    // Remove hidden elements
    // aria-hidden="true": only remove purely decorative elements (no interactive children,
    // no real text). Sites like Airbnb set aria-hidden on visually-rendered nav bars.
    for node in doc.select("[aria-hidden='true']").iter() {
        let inner = node.inner_html().to_string();
        let has_interactive = inner.contains("<a ")
            || inner.contains("<a>")
            || inner.contains("<button")
            || inner.contains("<input")
            || inner.contains("<select")
            || inner.contains("<textarea");
        let text = node.text().to_string();
        let has_text = text.split_whitespace().count() > 2;
        if !has_interactive && !has_text {
            node.remove();
        }
    }
    doc.select("[style*='display: none']").remove();
    doc.select("[style*='display:none']").remove();
    doc.select("[style*='visibility: hidden']").remove();
    doc.select("[style*='visibility:hidden']").remove();
    doc.select("[style*='opacity: 0']").remove();
    doc.select("[style*='opacity:0']").remove();

    // Remove JSON-LD and other data scripts
    doc.select("script[type='application/ld+json']").remove();
    doc.select("script[type='application/json']").remove();

    // Remove base64 inline images (placeholders/tracking pixels)
    for node in doc.select("img[src^='data:image/']").iter() {
        node.remove();
    }

    // Remove hidden <code> blocks (SPA JSON state, e.g. LinkedIn bpr-guid)
    for node in doc.select("code").iter() {
        let style = node.attr("style").unwrap_or_default().to_string();
        if style.contains("display:none") || style.contains("display: none") {
            node.remove();
        }
    }

    // Remove empty divs/spans with no text and no interactive children
    // (multiple passes to catch nested empties)
    for _ in 0..3 {
        for node in doc.select("div, span").iter() {
            let text = node.text().to_string();
            let inner = node.inner_html().to_string();
            let has_meaningful = inner.contains("<img")
                || inner.contains("<input")
                || inner.contains("<button")
                || inner.contains("<a ")
                || inner.contains("<select")
                || inner.contains("<textarea");
            if text.trim().is_empty() && !has_meaningful {
                node.remove();
            }
        }
    }

    // Strip useless attributes from all elements
    strip_attributes(&doc);

    doc.html().to_string()
}

/// Strip all attributes except semantically useful ones.
fn strip_attributes(doc: &Document) {
    for node in doc.select("*").iter() {
        let attrs: Vec<String> = node.attrs()
            .into_iter()
            .map(|a| a.name.local.to_string())
            .collect();
        for attr_name in &attrs {
            let s: &str = attr_name.as_str();
            // Strip data-* attributes (SPA state/JSON payloads)
            if s.starts_with("data-") {
                node.remove_attr(attr_name);
                continue;
            }
            if !KEEP_ATTRS.contains(&s) {
                node.remove_attr(attr_name);
            }
        }
    }
}

/// Extract all <img> tags from a Document.
fn extract_images(doc: &Document, page_url: &str) -> Vec<ImageRef> {
    let base_url = page_url.trim_end_matches('/');
    let mut images = vec![];

    for img in doc.select("img").iter() {
        let src = img.attr("src").unwrap_or_default().to_string();
        if src.is_empty() || src.starts_with("data:") {
            continue;
        }

        let url = resolve_url(&src, page_url, base_url);
        let alt = img.attr("alt").unwrap_or_default().to_string();
        let url_hash = crate::hasher::structural_hash("img", &[], &[&url]);

        images.push(ImageRef {
            url,
            alt,
            description: None,
            url_hash,
            description_tokens: 0,
        });
    }

    images
}

/// Attach images to nodes (distribute by node count for now).
fn attach_images_to_nodes(nodes: &mut [SemanticNode], mut images: Vec<ImageRef>) {
    if images.is_empty() || nodes.is_empty() {
        return;
    }

    let per_node = (images.len() / nodes.len()).max(1);

    for node in nodes.iter_mut() {
        let take = per_node.min(images.len());
        node.images.extend(images.drain(..take));
        if images.is_empty() {
            break;
        }
    }

    if !images.is_empty() {
        if let Some(last) = nodes.last_mut() {
            last.images.extend(images);
        }
    }
}

/// Compute the structural hash of raw HTML. Cheap way to detect page changes
/// without running the full pipeline.
pub fn structural_hash_of_html(html: &str) -> String {
    let cleaned = clean_html(html);
    let doc = Document::from(cleaned.as_str());
    compute_dom_structural_hash(&doc)
}

/// Build a structural Merkle hash from the DOM.
fn compute_dom_structural_hash(doc: &Document) -> String {
    let body = doc.select("body");
    if body.is_empty() {
        return hasher::structural_hash("html", &[], &[]);
    }

    let body_html = body.html().to_string();
    let body_doc = Document::from(body_html.as_str());

    hash_element_recursive(&body_doc, "body")
}

fn hash_element_recursive(doc: &Document, selector: &str) -> String {
    let el = doc.select(selector);
    if el.is_empty() {
        return hasher::structural_hash(selector, &[], &[]);
    }

    let tag = selector.split_whitespace().last().unwrap_or(selector);
    let classes_str = el.attr("class").unwrap_or_default().to_string();
    let classes: Vec<&str> = classes_str.split_whitespace().collect();

    let children_hashes: Vec<String> = el
        .select(":scope > *")
        .iter()
        .enumerate()
        .map(|(i, _child)| {
            let child_selector = format!("{selector} > :nth-child({})", i + 1);
            hash_element_recursive(doc, &child_selector)
        })
        .collect();

    let children_refs: Vec<&str> = children_hashes.iter().map(|s| s.as_str()).collect();
    hasher::structural_hash(tag, &classes, &children_refs)
}

fn extract_domain(url: &str) -> String {
    url.split("://")
        .nth(1)
        .unwrap_or(url)
        .split('/')
        .next()
        .unwrap_or(url)
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_domain() {
        assert_eq!(extract_domain("https://www.google.com/search?q=test"), "www.google.com");
        assert_eq!(extract_domain("http://example.com"), "example.com");
    }

    #[test]
    fn test_estimate_tokens() {
        assert_eq!(estimate_tokens("hello world"), 2);
        assert_eq!(estimate_tokens(""), 1);
    }

    #[test]
    fn test_clean_html() {
        let html = r#"<html><head><title>Test</title></head><body>
            <script>alert('xss')</script>
            <div class="content">Hello</div>
            <div class="hidden">Secret</div>
        </body></html>"#;
        let cleaned = clean_html(html);
        assert!(!cleaned.contains("alert"));
        assert!(cleaned.contains("Hello"));
        assert!(!cleaned.contains("Secret"));
    }

    #[test]
    fn test_chunk_dom_basic() {
        let html = r#"<html><body>
            <header><nav><a href="/">Home</a><a href="/about">About</a></nav></header>
            <main><div class="grid"><div class="item">A</div><div class="item">B</div><div class="item">C</div><div class="item">D</div><div class="item">E</div></div></main>
            <footer><a href="/terms">Terms</a></footer>
        </body></html>"#;
        let doc = Document::from(html);
        let chunks = chunk_dom(&doc, "https://example.com");
        assert_eq!(chunks.len(), 3, "should produce header, main, footer chunks");
        assert_eq!(chunks[0].tag, "header");
        assert_eq!(chunks[1].tag, "main");
        assert_eq!(chunks[2].tag, "footer");
    }
}
