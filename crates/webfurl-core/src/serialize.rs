use crate::tree::{SemanticNode, SemanticTree};
use crate::unfold::UnfoldState;

/// Serialize a SemanticTree into a compact text representation for the LLM context window.
/// Respects the current UnfoldState to show/hide children.
pub fn serialize_tree(
    tree: &SemanticTree,
    state: &UnfoldState,
    collapsed_pages: &[CollapsedPage],
) -> String {
    let mut out = String::new();

    out.push_str("[WEBFURL]\n");

    // Previously visited pages (collapsed summaries)
    for page in collapsed_pages {
        out.push_str(&format!(
            "[previously: {} - {}] ({} tokens saved)\n",
            page.domain, page.summary, page.original_tokens
        ));
    }

    // Current page
    out.push_str(&format!("[current: {}]\n", tree.url));

    for node in &tree.root_nodes {
        serialize_node(&mut out, node, state, 1);
    }

    out.push_str("[/WEBFURL]");
    out
}

fn serialize_node(
    out: &mut String,
    node: &SemanticNode,
    state: &UnfoldState,
    depth: usize,
) {
    let indent = "  ".repeat(depth);

    // Build the node line
    let mut line = format!("{indent}[{}]", node.summary);

    // Add node ID reference
    line.push_str(&format!(" {{#{}}}", node.id));

    // For interactive leaf nodes (single action, typically a button/link/input child),
    // show the action type so the AI knows what it can do with this node.
    if node.actions.len() == 1 {
        let tag = match &node.actions[0] {
            crate::actions::Action::Click { .. } => "clickable",
            crate::actions::Action::Fill { description, .. } => {
                // Show field hint for fill actions
                let _ = description; // used below
                "fillable"
            }
            crate::actions::Action::Select { .. } => "selectable",
            crate::actions::Action::Toggle { .. } => "toggleable",
        };
        line.push_str(&format!(" ({tag})"));
    } else if node.actions.len() > 1 {
        // Multiple actions on one node (legacy / parent prompts) — show count
        let mut types = std::collections::BTreeSet::new();
        for a in &node.actions {
            types.insert(match a {
                crate::actions::Action::Click { .. } => "clickable",
                crate::actions::Action::Fill { .. } => "fillable",
                crate::actions::Action::Select { .. } => "selectable",
                crate::actions::Action::Toggle { .. } => "toggleable",
            });
        }
        let joined: Vec<&str> = types.into_iter().collect();
        line.push_str(&format!(" ({})", joined.join(", ")));
    }

    out.push_str(&line);
    out.push('\n');

    // Images in this node — only show individually if vision-described, otherwise summarize
    if !node.images.is_empty() {
        let described: Vec<&str> = node.images.iter()
            .filter_map(|img| img.description.as_deref())
            .collect();
        let undescribed = node.images.len() - described.len();

        for desc in &described {
            out.push_str(&format!("{indent}  [img: {desc}]\n"));
        }
        if undescribed > 0 {
            out.push_str(&format!(
                "{indent}  [{undescribed} image{} — use describe #{} to inspect]\n",
                if undescribed > 1 { "s" } else { "" },
                node.id,
            ));
        }
    }

    // If this node is unfolded, render children
    if state.unfolded.contains(&node.id) && !node.children.is_empty() {
        for child in &node.children {
            serialize_node(out, child, state, depth + 1);
        }
    } else if !node.children.is_empty() {
        // Show a fold hint
        let child_count = node.children.len();
        let extra_tokens = node.unfold_cost();
        out.push_str(&format!(
            "{indent}  ... ({child_count} children, +{extra_tokens} tokens to unfold)\n"
        ));
    }

    // For leaf nodes with raw text, show it as unfoldable ground-truth data
    if node.children.is_empty() {
        if let Some(raw) = &node.raw_text {
            if state.unfolded.contains(&node.id) {
                out.push_str(&format!("{indent}  [raw] {raw}\n"));
            } else if node.raw_text_tokens > 0 {
                out.push_str(&format!(
                    "{indent}  ... (raw text, +{} tokens to unfold)\n",
                    node.raw_text_tokens
                ));
            }
        }
    }
}

/// A collapsed representation of a previously visited page.
#[derive(Debug, Clone)]
pub struct CollapsedPage {
    pub url: String,
    pub domain: String,
    pub summary: String,
    pub original_tokens: u32,
}

/// Collapse a SemanticTree into a short summary for conversation history.
/// This is called when navigating away from a page.
pub fn collapse_tree(tree: &SemanticTree, interaction_summary: &str) -> CollapsedPage {
    let top_level: Vec<&str> = tree
        .root_nodes
        .iter()
        .map(|n| n.summary.as_str())
        .collect();

    let summary = if interaction_summary.is_empty() {
        format!("Visited. Sections: {}", top_level.join(", "))
    } else {
        format!("{interaction_summary}. Sections: {}", top_level.join(", "))
    };

    CollapsedPage {
        url: tree.url.clone(),
        domain: tree.domain.clone(),
        summary,
        original_tokens: tree.full_token_count,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::{SemanticNode, SemanticTree};
    use crate::unfold::{initial_pack, unfold_node};

    fn make_test_tree() -> SemanticTree {
        SemanticTree {
            url: "https://example.com".into(),
            domain: "example.com".into(),
            title: "Example".into(),
            root_nodes: vec![
                SemanticNode {
                    id: "nav".into(),
                    summary: "Navigation: Home, About, Contact".into(),
                    embedding: vec![],
                    structural_hash: "h1".into(),
                    is_dynamic: false,
                    dynamic_selector: None,
                    children: vec![],
                    actions: vec![crate::actions::Action::Click {
                        selector: "nav a".into(),
                        description: "Navigate".into(),
                    }],
                    images: vec![],
                    dom_selector: "nav".into(),
                    token_count: 8,
                    subtree_token_count: 8,
                    stable: false,
                    raw_text: None,
                    raw_text_tokens: 0,
                },
                SemanticNode {
                    id: "main".into(),
                    summary: "Main content with search and results".into(),
                    embedding: vec![],
                    structural_hash: "h2".into(),
                    is_dynamic: false,
                    dynamic_selector: None,
                    children: vec![
                        SemanticNode {
                            id: "main.search".into(),
                            summary: "Search box".into(),
                            embedding: vec![],
                            structural_hash: "h3".into(),
                            is_dynamic: false,
                            dynamic_selector: None,
                            children: vec![],
                            actions: vec![crate::actions::Action::Fill {
                                selector: "#search".into(),
                                field_type: crate::actions::FieldType::Search,
                                description: "Search input".into(),
                            }],
                            images: vec![],
                            dom_selector: "#search-box".into(),
                            token_count: 4,
                            subtree_token_count: 4,
                            stable: false,
                            raw_text: None,
                            raw_text_tokens: 0,
                        },
                        SemanticNode {
                            id: "main.results".into(),
                            summary: "10 search results".into(),
                            embedding: vec![],
                            structural_hash: "h4".into(),
                            is_dynamic: true,
                            dynamic_selector: Some(".results".into()),
                            children: vec![],
                            actions: vec![],
                            images: vec![],
                            dom_selector: ".results".into(),
                            token_count: 6,
                            subtree_token_count: 6,
                            stable: false,
                            raw_text: None,
                            raw_text_tokens: 0,
                        },
                    ],
                    actions: vec![],
                    images: vec![],
                    dom_selector: "main".into(),
                    token_count: 10,
                    subtree_token_count: 20,
                    stable: false,
                    raw_text: None,
                    raw_text_tokens: 0,
                },
            ],
            compressed_token_count: 18,
            full_token_count: 28,
            structural_hash: "page".into(),
            created_at: 0,
            dynamic_slots_filled_at: 0,
        }
    }

    #[test]
    fn test_serialize_folded() {
        let tree = make_test_tree();
        let state = initial_pack(&tree, 5000, 128_000);
        let output = serialize_tree(&tree, &state, &[]);
        assert!(output.contains("[WEBFURL]"));
        assert!(output.contains("[/WEBFURL]"));
        assert!(output.contains("Navigation: Home, About, Contact"));
        assert!(output.contains("... (2 children"));
    }

    #[test]
    fn test_serialize_unfolded() {
        let tree = make_test_tree();
        let mut state = initial_pack(&tree, 5000, 128_000);
        unfold_node(&tree, &mut state, "main");
        let output = serialize_tree(&tree, &state, &[]);
        assert!(output.contains("Search box"));
        assert!(output.contains("10 search results"));
    }

    #[test]
    fn test_serialize_with_collapsed_pages() {
        let tree = make_test_tree();
        let state = initial_pack(&tree, 5000, 128_000);
        let collapsed = vec![CollapsedPage {
            url: "https://google.com".into(),
            domain: "google.com".into(),
            summary: "Searched for flights".into(),
            original_tokens: 5000,
        }];
        let output = serialize_tree(&tree, &state, &collapsed);
        assert!(output.contains("previously: google.com - Searched for flights"));
    }

    #[test]
    fn test_collapse_tree() {
        let tree = make_test_tree();
        let collapsed = collapse_tree(&tree, "Searched for product X");
        assert!(collapsed.summary.contains("Searched for product X"));
        assert_eq!(collapsed.original_tokens, 28);
    }
}
