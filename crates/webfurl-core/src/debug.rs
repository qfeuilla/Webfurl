use crate::tree::{SemanticNode, SemanticTree};
use crate::unfold::UnfoldState;
use crate::serialize::CollapsedPage;

const RESET: &str = "\x1b[0m";
const BOLD: &str = "\x1b[1m";
const DIM: &str = "\x1b[2m";
const RED: &str = "\x1b[31m";
const GREEN: &str = "\x1b[32m";
const YELLOW: &str = "\x1b[33m";
const BLUE: &str = "\x1b[34m";
const MAGENTA: &str = "\x1b[35m";
const CYAN: &str = "\x1b[36m";
const WHITE: &str = "\x1b[37m";
const BG_DARK: &str = "\x1b[48;5;236m";

/// Print a full debug view of the semantic tree structure.
pub fn print_tree(tree: &SemanticTree) {
    println!("\n{BOLD}{CYAN}╔══════════════════════════════════════════════════╗{RESET}");
    println!("{BOLD}{CYAN}║  WEBFURL TREE: {}{RESET}", truncate(&tree.title, 34));
    println!("{BOLD}{CYAN}╠══════════════════════════════════════════════════╣{RESET}");
    println!("{CYAN}║{RESET} {DIM}URL:{RESET}  {}", tree.url);
    println!("{CYAN}║{RESET} {DIM}Hash:{RESET} {}", truncate(&tree.structural_hash, 16));
    println!("{CYAN}║{RESET} {DIM}Nodes:{RESET} {}", count_nodes(tree));
    println!("{CYAN}║{RESET} {DIM}Full tokens:{RESET}       {YELLOW}{}{RESET}", tree.full_token_count);
    println!("{CYAN}║{RESET} {DIM}Compressed tokens:{RESET} {GREEN}{}{RESET}", tree.compressed_token_count);
    println!(
        "{CYAN}║{RESET} {DIM}Compression:{RESET}       {BOLD}{GREEN}{:.0}x{RESET}",
        1.0 / tree.compression_ratio().max(0.001)
    );
    println!("{BOLD}{CYAN}╠══════════════════════════════════════════════════╣{RESET}");

    for node in &tree.root_nodes {
        print_node(node, 0, &[]);
    }

    println!("{BOLD}{CYAN}╚══════════════════════════════════════════════════╝{RESET}\n");
}

fn print_node(node: &SemanticNode, depth: usize, unfolded: &[String]) {
    let indent = "  ".repeat(depth);
    let connector = if depth == 0 { "╟─" } else { "├─" };
    let is_unfolded = unfolded.contains(&node.id);

    // Node ID and summary
    let dynamic_tag = if node.is_dynamic {
        format!(" {MAGENTA}⚡dynamic{RESET}")
    } else {
        String::new()
    };

    let action_tag = if !node.actions.is_empty() {
        let types: Vec<&str> = node.actions.iter().map(|a| match a {
            crate::actions::Action::Click { .. } => "🖱click",
            crate::actions::Action::Fill { .. } => "✏fill",
            crate::actions::Action::Select { .. } => "📋select",
            crate::actions::Action::Toggle { .. } => "🔘toggle",
        }).collect();
        format!(" {YELLOW}[{}]{RESET}", types.join(","))
    } else {
        String::new()
    };

    let fold_icon = if node.children.is_empty() {
        format!("{DIM}○{RESET}")
    } else if is_unfolded {
        format!("{GREEN}▼{RESET}")
    } else {
        format!("{YELLOW}▶{RESET}")
    };

    let token_info = format!(
        "{DIM}({} tok, {} subtree){RESET}",
        node.token_count, node.subtree_token_count
    );

    let emb_indicator = if node.embedding.is_empty() {
        format!("{DIM}∅{RESET}")
    } else {
        format!("{GREEN}⬡{RESET}")
    };

    println!(
        "{CYAN}║{RESET} {indent}{connector} {fold_icon} {emb_indicator} {BOLD}#{}{RESET} {token_info}{dynamic_tag}{action_tag}",
        node.id,
    );
    println!(
        "{CYAN}║{RESET} {indent}   {BG_DARK}{WHITE} {} {RESET}",
        truncate(&node.summary, 60)
    );

    if !node.dom_selector.is_empty() {
        println!(
            "{CYAN}║{RESET} {indent}   {DIM}selector: {}{RESET}",
            truncate(&node.dom_selector, 50)
        );
    }

    // Show images
    for img in &node.images {
        let status = if img.description.is_some() {
            format!("{GREEN}✓ described{RESET}")
        } else {
            format!("{YELLOW}? undescribed{RESET}")
        };
        let alt = if img.alt.is_empty() { "" } else { &img.alt };
        println!(
            "{CYAN}║{RESET} {indent}   🖼  {status} {DIM}{}{RESET} {DIM}\"{alt}\"{RESET}",
            truncate(&img.url, 40)
        );
    }

    for child in &node.children {
        print_node(child, depth + 1, unfolded);
    }
}

/// Print the current unfold state and budget usage.
pub fn print_unfold_state(state: &UnfoldState, tree: &SemanticTree) {
    println!("\n{BOLD}{BLUE}┌─ UNFOLD STATE ───────────────────────────────────┐{RESET}");

    // Initial budget bar
    let initial_pct = ((state.token_usage as f32 / state.initial_budget as f32) * 100.0).min(999.0) as u32;
    let bar_width = 30;
    let filled = (initial_pct as usize * bar_width / 100).min(bar_width);
    let bar: String = "█".repeat(filled) + &"░".repeat(bar_width - filled);

    let bar_color = if initial_pct > 100 { RED } else if initial_pct > 70 { YELLOW } else { GREEN };
    let expanded = if !state.within_initial_budget() {
        format!(" {RED}(expanded beyond initial){RESET}")
    } else {
        String::new()
    };

    println!(
        "{BLUE}│{RESET} Initial furl: {bar_color}{bar}{RESET} {BOLD}{}/{}{RESET} tokens ({initial_pct}%){expanded}",
        state.token_usage, state.initial_budget
    );
    println!(
        "{BLUE}│{RESET} Max session:  {BOLD}{}/{}{RESET} tokens",
        state.token_usage, state.max_budget
    );
    println!(
        "{BLUE}│{RESET} Remaining:    {GREEN}{}{RESET} tokens (hard max)",
        state.remaining_budget()
    );

    if !state.unfolded.is_empty() {
        println!("{BLUE}│{RESET} {BOLD}Unfolded nodes:{RESET}");
        for id in &state.unfolded {
            if let Some(node) = tree.find_node(id) {
                println!(
                    "{BLUE}│{RESET}   {GREEN}▼{RESET} #{id} (+{} tokens) - {}",
                    node.unfold_cost(),
                    truncate(&node.summary, 40)
                );
            }
        }
    } else {
        println!("{BLUE}│{RESET} {DIM}No nodes unfolded (all folded){RESET}");
    }

    println!("{BOLD}{BLUE}└──────────────────────────────────────────────────┘{RESET}\n");
}

/// Print the context window as it would be sent to the LLM, with annotations.
pub fn print_context_window(
    tree: &SemanticTree,
    state: &UnfoldState,
    collapsed_pages: &[CollapsedPage],
    user_query: Option<&str>,
) {
    let context = crate::serialize::serialize_tree(tree, state, collapsed_pages);
    let approx_tokens = context.len() / 4;

    println!("\n{BOLD}{MAGENTA}┌─ CONTEXT WINDOW (as sent to LLM) ────────────────┐{RESET}");
    println!("{MAGENTA}│{RESET} {DIM}Approx tokens: ~{approx_tokens}{RESET}");
    println!("{MAGENTA}├──────────────────────────────────────────────────┤{RESET}");

    for line in context.lines() {
        let colored = colorize_context_line(line);
        println!("{MAGENTA}│{RESET} {colored}");
    }

    if let Some(query) = user_query {
        println!("{MAGENTA}├──────────────────────────────────────────────────┤{RESET}");
        println!("{MAGENTA}│{RESET} {BOLD}{WHITE}User: {query}{RESET}");
    }

    println!("{BOLD}{MAGENTA}└──────────────────────────────────────────────────┘{RESET}\n");
}

fn colorize_context_line(line: &str) -> String {
    let trimmed = line.trim();
    if trimmed.starts_with("[WEBFURL]") || trimmed.starts_with("[/WEBFURL]") {
        format!("{BOLD}{CYAN}{line}{RESET}")
    } else if trimmed.starts_with("[previously:") {
        format!("{DIM}{line}{RESET}")
    } else if trimmed.starts_with("[current:") {
        format!("{BOLD}{GREEN}{line}{RESET}")
    } else if trimmed.contains("*dynamic*") {
        format!("{MAGENTA}{line}{RESET}")
    } else if trimmed.contains("... (") {
        format!("{YELLOW}{line}{RESET}")
    } else if trimmed.contains("{#") {
        // Highlight node IDs
        let colored = line.replace("{#", &format!("{CYAN}{{#"));
        let colored = colored.replace('}', &format!("}}{RESET}"));
        colored
    } else {
        line.to_string()
    }
}

/// Print compression stats comparison.
pub fn print_compression_stats(tree: &SemanticTree) {
    let total_nodes = count_nodes(tree);
    let dynamic_nodes = count_dynamic_nodes(tree);
    let action_nodes = count_action_nodes(tree);
    let max_depth = max_tree_depth(tree);

    println!("\n{BOLD}{GREEN}┌─ COMPRESSION STATS ──────────────────────────────┐{RESET}");
    println!("{GREEN}│{RESET}  {DIM}Full page tokens:{RESET}     {RED}{:>6}{RESET}", tree.full_token_count);
    println!("{GREEN}│{RESET}  {DIM}Compressed tokens:{RESET}    {GREEN}{:>6}{RESET}", tree.compressed_token_count);
    println!(
        "{GREEN}│{RESET}  {DIM}Compression ratio:{RESET}    {BOLD}{GREEN}{:>5.0}x{RESET}",
        1.0 / tree.compression_ratio().max(0.001)
    );
    let image_count = count_images(tree);
    let described_images = count_described_images(tree);
    println!("{GREEN}│{RESET}  {DIM}Total nodes:{RESET}          {:>6}", total_nodes);
    println!("{GREEN}│{RESET}  {DIM}Dynamic nodes:{RESET}        {MAGENTA}{:>6}{RESET}", dynamic_nodes);
    println!("{GREEN}│{RESET}  {DIM}Interactive nodes:{RESET}    {YELLOW}{:>6}{RESET}", action_nodes);
    println!("{GREEN}│{RESET}  {DIM}Images:{RESET}               {:>6} ({described_images} described)", image_count);
    println!("{GREEN}│{RESET}  {DIM}Max depth:{RESET}            {:>6}", max_depth);
    println!("{GREEN}│{RESET}  {DIM}Structural hash:{RESET}      {}", truncate(&tree.structural_hash, 20));
    println!("{BOLD}{GREEN}└──────────────────────────────────────────────────┘{RESET}\n");
}

fn count_nodes(tree: &SemanticTree) -> usize {
    tree.root_nodes.iter().map(count_nodes_recursive).sum()
}

fn count_nodes_recursive(node: &SemanticNode) -> usize {
    1 + node.children.iter().map(count_nodes_recursive).sum::<usize>()
}

fn count_dynamic_nodes(tree: &SemanticTree) -> usize {
    tree.all_nodes().iter().filter(|n| n.is_dynamic).count()
}

fn count_action_nodes(tree: &SemanticTree) -> usize {
    tree.all_nodes().iter().filter(|n| !n.actions.is_empty()).count()
}

fn count_images(tree: &SemanticTree) -> usize {
    tree.all_nodes().iter().map(|n| n.images.len()).sum()
}

fn count_described_images(tree: &SemanticTree) -> usize {
    tree.all_nodes().iter().flat_map(|n| &n.images).filter(|i| i.description.is_some()).count()
}

fn max_tree_depth(tree: &SemanticTree) -> usize {
    tree.root_nodes.iter().map(|n| node_depth(n)).max().unwrap_or(0)
}

fn node_depth(node: &SemanticNode) -> usize {
    if node.children.is_empty() {
        1
    } else {
        1 + node.children.iter().map(|c| node_depth(c)).max().unwrap_or(0)
    }
}

fn truncate(s: &str, max_len: usize) -> &str {
    if s.len() <= max_len {
        s
    } else {
        // Find the last char boundary at or before max_len to avoid
        // panicking on multi-byte UTF-8 characters (e.g. en-dash '–').
        let mut end = max_len.min(s.len());
        while end > 0 && !s.is_char_boundary(end) {
            end -= 1;
        }
        &s[..end]
    }
}
