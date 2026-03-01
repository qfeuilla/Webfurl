# WebFurl

A Rust browser agent that uses a **recursively compressible semantic representation** of web pages to minimize LLM context usage. The representation is dynamically cached across users, query-aware, and fully linked to the live DOM for real browser interactions.

## How it works

WebFurl compresses a full web page (often 200k+ tokens of raw HTML) into a hierarchical semantic tree (typically 20-50 tokens at the top level). The agent can then **unfold** parts of the tree on demand, spending context budget only on what matters.

**Compression pipeline:**
1. Raw HTML is chunked at semantic boundaries (header, nav, main, sections, grids)
2. Leaf chunks are compressed in parallel via LLM calls
3. Parent nodes get structural summaries from child summaries (bottom-up)
4. Interactive elements (links, buttons, inputs) are extracted from the raw DOM with stable CSS selectors
5. Everything is cached by content hash in MongoDB, so unchanged subtrees are never recompressed

**What makes it different:**
- **Recursive compression** — a page is a tree, not a flat summary. You can zoom into any branch.
- **Cross-user cache** — the static parts of airbnb.com are compressed once and reused by everyone. Only dynamic content (prices, availability) gets recompressed.
- **Query-driven unfolding** — when the user asks "find me a cheap listing", the tree auto-unfolds the most relevant nodes using embedding similarity, so the LLM sees a focused view without wasting budget on irrelevant sections.
- **DOM-linked actions** — every interactive element has a pre-computed CSS selector that works against the live browser DOM. The agent can click links, fill forms, and navigate, with automatic handling of new tabs and page loads.
- **Vision support** — images in the tree can be described on demand via a vision model, with descriptions cached.

## Prerequisites

- **Rust** (stable, 1.75+) — [rustup.rs](https://rustup.rs)
- **Docker** — for MongoDB ([Docker Desktop](https://docker.com/products/docker-desktop) on Mac)
- **Chrome or Chromium** — auto-detected on Mac and Linux
- **OpenRouter API key** — for LLM calls ([openrouter.ai](https://openrouter.ai))

## Quick start

```bash
# Clone
git clone <repo-url>
cd Webfurl

# Configure
cp .env.example .env
# Edit .env and add your OPENROUTER_API_KEY

# Run (builds, starts MongoDB, launches browser + agent)
./start.sh
```

That's it. The start script:
1. Builds the Rust workspace (`cargo build --release`)
2. Starts MongoDB in Docker (creates container if needed)
3. Launches the API server on `:3001`
4. Opens an interactive agent session with a visible Chrome browser

## Usage

Once running, you'll see an interactive prompt:

```
/url https://www.airbnb.com/s/Mountain-View--CA/homes
```

The agent compresses the page into a semantic tree, then you can chat naturally:

```
> Find me the cheapest listing with good reviews
```

The agent will:
1. Pre-unfold relevant nodes (query-driven, using embeddings)
2. Read the compressed page context
3. Click on listings, navigate pages, fill search forms
4. Report back with findings

### Commands

| Command | Description |
|---------|-------------|
| `/url <url>` | Navigate to a URL |
| `/unfold <node_id>` | Manually expand a tree node |
| `/fold <node_id>` | Collapse a node back |
| `/search <query>` | Semantic search — unfolds the most relevant nodes |
| `/tree` | Print the current tree structure |
| `/screenshot` | Full page screenshot |
| `/screenshot <selector>` | Element screenshot |
| `/browser` | Open the current page in your default browser |
| `/quit` | Exit |

Or just type naturally — the agent handles navigation, clicking, and form filling autonomously.

## Configuration

All configuration is in `.env`:

| Variable | Description | Example |
|----------|-------------|---------|
| `OPENROUTER_API_KEY` | **Required.** Your OpenRouter API key | `sk-or-v1-...` |
| `WEBFURL_COMPRESSION_MODEL` | LLM for page compression | `openai/gpt-oss-120b` |
| `WEBFURL_AGENT_MODEL` | LLM for the agent | `anthropic/claude-sonnet-4.6` |
| `WEBFURL_VISION_MODEL` | Vision model for images | `google/gemini-2.5-flash` |
| `WEBFURL_TOKEN_BUDGET` | Initial context budget per page (tokens) | `5000` |
| `WEBFURL_MAX_BUDGET` | Hard ceiling the agent can expand to | `128000` |
| `CHROME_PATH` | Chrome binary path (auto-detected) | `/usr/bin/google-chrome` |
| `WEBFURL_HEADLESS` | Set to `1` for headless mode | `1` |
| `MONGODB_URI` | MongoDB connection string | `mongodb://localhost:27017` |

## Architecture

```
Webfurl/
├── crates/
│   ├── webfurl-core/        # Compression pipeline, tree, cache, unfold, serialize
│   ├── webfurl-agent/       # Browser agent, Chrome CDP, interactive CLI
│   └── webfurl-server/      # Axum API server (REST endpoints)
├── start.sh                 # One-command launcher
├── stop.sh                  # Cleanup
└── .env.example
```

**Core modules** (`webfurl-core`):
- `pipeline.rs` — HTML → SemanticTree (DOM chunking, parallel LLM compression, interactive element extraction)
- `tree.rs` — SemanticNode / SemanticTree data structures
- `unfold.rs` — Budget-based unfolding, semantic query unfold with ancestor chain resolution
- `serialize.rs` — Tree → `[WEBFURL]` text block for LLM context
- `cache.rs` — MongoDB content-hash cache (cross-user, chunk-level)
- `embeddings.rs` — OpenRouter embedding client (Qwen3-Embedding-8B)

**Agent** (`webfurl-agent`):
- `agent.rs` — Conversation loop, action execution, query-driven pre-unfolding
- `browser.rs` — Chrome CDP session (navigation, click, fill, tab management, page load detection)

## How the cache works

Every chunk of HTML is hashed by content. When any user visits a page:
- Static chunks (nav, footer, layout) → cache hit, zero LLM calls
- Dynamic chunks (prices, dates, user-specific content) → recompressed

This means the first user to visit airbnb.com pays the full compression cost. The second user compressing the same page layout pays only for the dynamic parts. The cache is stored in MongoDB and persists across sessions.

## License

AGPL-3.0 — see [LICENSE](LICENSE)
