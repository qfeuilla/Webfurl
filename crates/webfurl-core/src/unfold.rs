use std::collections::BinaryHeap;
use std::cmp::Ordering;

use tracing::{debug, warn};

use crate::embeddings::cosine_similarity;
use crate::tree::SemanticTree;

/// A candidate node for unfolding, scored by relevance.
#[derive(Debug)]
struct UnfoldCandidate {
    node_id: String,
    score: f32,
    unfold_cost: u32,
}

impl PartialEq for UnfoldCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}

impl Eq for UnfoldCandidate {}

impl PartialOrd for UnfoldCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for UnfoldCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .partial_cmp(&other.score)
            .unwrap_or(Ordering::Equal)
    }
}

/// Tracks which nodes are currently unfolded in the context window.
///
/// Budget model:
/// - `initial_budget`: tokens used for the first auto-unfold pass per page
/// - `max_budget`: hard ceiling the agent can expand up to (across all pages in session)
/// - The agent starts at `initial_budget` and can request more via UNFOLD/SEARCH up to `max_budget`
#[derive(Debug, Clone)]
pub struct UnfoldState {
    /// Node IDs that are currently unfolded (showing children instead of summary)
    pub unfolded: Vec<String>,
    /// Current total token usage for the current page
    pub token_usage: u32,
    /// Initial per-page budget (auto-unfold fills up to this)
    pub initial_budget: u32,
    /// Hard max the agent can expand to
    pub max_budget: u32,
}

impl UnfoldState {
    pub fn new(initial_budget: u32, max_budget: u32) -> Self {
        Self {
            unfolded: vec![],
            token_usage: 0,
            initial_budget,
            max_budget,
        }
    }

    /// Remaining tokens before hitting the hard max
    pub fn remaining_budget(&self) -> u32 {
        self.max_budget.saturating_sub(self.token_usage)
    }

    /// Remaining tokens in the initial auto-unfold budget
    pub fn remaining_initial_budget(&self) -> u32 {
        self.initial_budget.saturating_sub(self.token_usage)
    }

    /// Whether we're still within the initial budget (vs expanded by agent)
    pub fn within_initial_budget(&self) -> bool {
        self.token_usage <= self.initial_budget
    }
}

/// Initial packing: fit as much of the tree's root-level summaries as possible.
pub fn initial_pack(tree: &SemanticTree, initial_budget: u32, max_budget: u32) -> UnfoldState {
    let mut state = UnfoldState::new(initial_budget, max_budget);
    state.token_usage = tree.compressed_token_count;
    state
}

/// Unfold a specific node by ID. Returns the additional tokens consumed.
pub fn unfold_node(
    tree: &SemanticTree,
    state: &mut UnfoldState,
    node_id: &str,
) -> Option<u32> {
    if state.unfolded.contains(&node_id.to_string()) {
        debug!(node_id, "unfold_node: already unfolded, skipping");
        return Some(0);
    }

    let node = tree.find_node(node_id)?;
    if !node.is_foldable() {
        debug!(node_id, "unfold_node: node has no children, nothing to unfold");
        return Some(0);
    }

    let cost = node.unfold_cost();
    if state.token_usage + cost > state.max_budget {
        warn!(
            node_id,
            cost,
            token_usage = state.token_usage,
            max_budget = state.max_budget,
            "unfold_node: BUDGET EXCEEDED, refusing to unfold"
        );
        return None;
    }

    debug!(node_id, cost, "unfold_node: unfolding");
    state.unfolded.push(node_id.to_string());
    state.token_usage += cost;
    Some(cost)
}

/// Fold a previously unfolded node. Returns the tokens reclaimed.
pub fn fold_node(
    tree: &SemanticTree,
    state: &mut UnfoldState,
    node_id: &str,
) -> Option<u32> {
    let pos = state.unfolded.iter().position(|id| id == node_id)?;
    let node = tree.find_node(node_id)?;
    let reclaimed = node.unfold_cost();
    state.unfolded.remove(pos);
    state.token_usage = state.token_usage.saturating_sub(reclaimed);
    Some(reclaimed)
}

/// Semantic query unfolding: given a query embedding, find and unfold
/// the most relevant nodes that fit within the remaining budget.
pub fn semantic_unfold(
    tree: &SemanticTree,
    state: &mut UnfoldState,
    query_embedding: &[f32],
    max_unfolds: usize,
) -> Vec<String> {
    let mut heap = BinaryHeap::new();

    for node in tree.all_nodes() {
        if !node.is_foldable() {
            continue;
        }
        if state.unfolded.contains(&node.id) {
            continue;
        }
        if node.embedding.is_empty() {
            continue;
        }

        let relevance = cosine_similarity(&node.embedding, query_embedding);
        let cost = node.unfold_cost();
        if cost == 0 {
            continue;
        }

        // Score: relevance weighted by information density (more children tokens per cost = better)
        let density = node.subtree_token_count as f32 / cost as f32;
        let score = relevance * (1.0 + density.ln().max(0.0));

        heap.push(UnfoldCandidate {
            node_id: node.id.clone(),
            score,
            unfold_cost: cost,
        });
    }

    let mut unfolded_ids = vec![];
    let mut count = 0;

    while let Some(candidate) = heap.pop() {
        if count >= max_unfolds {
            break;
        }

        // Calculate total cost: the target node PLUS all its ancestors that aren't unfolded yet.
        // Without unfolding ancestors, a deep node would be invisible in the serialized output.
        let ancestors = tree.ancestor_path(&candidate.node_id).unwrap_or_default();
        let mut total_cost = candidate.unfold_cost;
        let mut ancestors_to_unfold = vec![];
        for ancestor_id in &ancestors {
            if !state.unfolded.contains(ancestor_id) {
                if let Some(ancestor_node) = tree.find_node(ancestor_id) {
                    let ancestor_cost = ancestor_node.unfold_cost();
                    total_cost += ancestor_cost;
                    ancestors_to_unfold.push((ancestor_id.clone(), ancestor_cost));
                }
            }
        }

        if state.token_usage + total_cost > state.max_budget {
            continue;
        }

        // Unfold ancestors first (root → parent order)
        for (anc_id, anc_cost) in &ancestors_to_unfold {
            state.unfolded.push(anc_id.clone());
            state.token_usage += anc_cost;
            unfolded_ids.push(anc_id.clone());
            debug!(ancestor = %anc_id, cost = anc_cost, "unfold ancestor to reveal target");
        }

        // Unfold the target node itself
        state.unfolded.push(candidate.node_id.clone());
        state.token_usage += candidate.unfold_cost;
        unfolded_ids.push(candidate.node_id);
        count += 1;
    }

    unfolded_ids
}

/// Auto-unfold: only unfold root-level nodes (depth 0) to show the page overview.
/// Deeper nodes stay collapsed — the LLM can request unfolding as needed.
pub fn auto_unfold(
    tree: &SemanticTree,
    state: &mut UnfoldState,
) -> Vec<String> {
    let mut unfolded_ids = vec![];

    for node in &tree.root_nodes {
        if !node.is_foldable() || state.unfolded.contains(&node.id) {
            continue;
        }

        let cost = node.unfold_cost();
        if cost == 0 {
            continue;
        }

        if state.token_usage + cost > state.initial_budget {
            continue;
        }

        state.unfolded.push(node.id.clone());
        state.token_usage += cost;
        unfolded_ids.push(node.id.clone());
    }

    unfolded_ids
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::SemanticTree;

    fn make_test_tree() -> SemanticTree {
        SemanticTree {
            url: "https://example.com".into(),
            domain: "example.com".into(),
            title: "Test".into(),
            root_nodes: vec![
                SemanticNode {
                    id: "nav".into(),
                    summary: "Navigation bar".into(),
                    embedding: vec![],
                    structural_hash: "h1".into(),
                    is_dynamic: false,
                    dynamic_selector: None,
                    children: vec![],
                    actions: vec![],
                    images: vec![],
                    dom_selector: "nav".into(),
                    token_count: 5,
                    subtree_token_count: 5,
                    stable: false,
                    raw_text: None,
                    raw_text_tokens: 0,
                },
                SemanticNode {
                    id: "main".into(),
                    summary: "Main content area".into(),
                    embedding: vec![],
                    structural_hash: "h2".into(),
                    is_dynamic: false,
                    dynamic_selector: None,
                    children: vec![
                        SemanticNode {
                            id: "main.products".into(),
                            summary: "Product grid with 12 items".into(),
                            embedding: vec![],
                            structural_hash: "h3".into(),
                            is_dynamic: true,
                            dynamic_selector: Some(".products".into()),
                            children: vec![],
                            actions: vec![],
                            images: vec![],
                            dom_selector: ".products".into(),
                            token_count: 10,
                            subtree_token_count: 10,
                            stable: false,
                            raw_text: None,
                            raw_text_tokens: 0,
                        },
                    ],
                    actions: vec![],
                    images: vec![],
                    dom_selector: "main".into(),
                    token_count: 8,
                    subtree_token_count: 18,
                    stable: false,
                    raw_text: None,
                    raw_text_tokens: 0,
                },
            ],
            compressed_token_count: 13,
            full_token_count: 23,
            structural_hash: "page_hash".into(),
            created_at: 0,
            dynamic_slots_filled_at: 0,
        }
    }

    use crate::tree::SemanticNode;

    #[test]
    fn test_initial_pack() {
        let tree = make_test_tree();
        let state = initial_pack(&tree, 5000, 128_000);
        assert_eq!(state.token_usage, 13);
        assert_eq!(state.remaining_budget(), 128_000 - 13);
    }

    #[test]
    fn test_unfold_node() {
        let tree = make_test_tree();
        let mut state = initial_pack(&tree, 5000, 128_000);
        let cost = unfold_node(&tree, &mut state, "main");
        assert_eq!(cost, Some(10));
        assert_eq!(state.token_usage, 23);
    }

    #[test]
    fn test_unfold_over_budget() {
        let tree = make_test_tree();
        let mut state = initial_pack(&tree, 15, 15);
        let cost = unfold_node(&tree, &mut state, "main");
        assert_eq!(cost, None);
    }

    #[test]
    fn test_fold_node() {
        let tree = make_test_tree();
        let mut state = initial_pack(&tree, 5000, 128_000);
        unfold_node(&tree, &mut state, "main");
        assert_eq!(state.token_usage, 23);
        let reclaimed = fold_node(&tree, &mut state, "main");
        assert_eq!(reclaimed, Some(10));
        assert_eq!(state.token_usage, 13);
    }
}
