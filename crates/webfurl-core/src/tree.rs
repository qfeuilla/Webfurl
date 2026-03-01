use serde::{Deserialize, Serialize};

use crate::actions::Action;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageRef {
    /// Image source URL
    pub url: String,
    /// Alt text from HTML (may be empty)
    #[serde(default)]
    pub alt: String,
    /// Vision-model-generated description (cached across users)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Hash of the image URL for cache lookups
    pub url_hash: String,
    /// Estimated token cost of the description
    #[serde(default)]
    pub description_tokens: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticNode {
    /// Stable path-like ID: "nav", "main.products.3"
    pub id: String,
    /// Compressed text summary at this level
    pub summary: String,
    /// Embedding vector for semantic search
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub embedding: Vec<f32>,
    /// Merkle hash of the subtree structure (tags + classes + child hashes)
    pub structural_hash: String,
    /// Whether this node's text content changes across visits
    pub is_dynamic: bool,
    /// CSS selector to extract live content for dynamic nodes
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dynamic_selector: Option<String>,
    /// Child nodes (the unfolded version)
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub children: Vec<SemanticNode>,
    /// Interactive elements within this node
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub actions: Vec<Action>,
    /// Images found in this node's DOM region
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub images: Vec<ImageRef>,
    /// CSS selector that maps this node back to real DOM
    pub dom_selector: String,
    /// Token count for this node's summary alone
    pub token_count: u32,
    /// Total tokens if entire subtree is unrolled
    pub subtree_token_count: u32,
    /// If true, summary describes structure/type only (e.g. "grid of listing cards"),
    /// not specific content. Stable nodes are NOT recompressed when children change.
    #[serde(default)]
    pub stable: bool,
    /// Original text content for leaf nodes (no LLM hallucination risk).
    /// Shown when the node is fully unfolded to the deepest level.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_text: Option<String>,
    /// Token count of raw_text (if present)
    #[serde(default)]
    pub raw_text_tokens: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticTree {
    pub url: String,
    pub domain: String,
    pub title: String,
    pub root_nodes: Vec<SemanticNode>,
    /// Sum of all root summaries token counts
    pub compressed_token_count: u32,
    /// Total tokens if everything is unrolled
    pub full_token_count: u32,
    /// Page-level structural hash
    pub structural_hash: String,
    pub created_at: u64,
    pub dynamic_slots_filled_at: u64,
}

impl SemanticNode {
    /// Recursively find a node by ID
    pub fn find(&self, id: &str) -> Option<&SemanticNode> {
        if self.id == id {
            return Some(self);
        }
        for child in &self.children {
            if let Some(found) = child.find(id) {
                return Some(found);
            }
        }
        None
    }

    /// Recursively find a mutable node by ID
    pub fn find_mut(&mut self, id: &str) -> Option<&mut SemanticNode> {
        if self.id == id {
            return Some(self);
        }
        for child in &mut self.children {
            if let Some(found) = child.find_mut(id) {
                return Some(found);
            }
        }
        None
    }

    /// Collect all nodes depth-first
    pub fn iter_depth_first(&self) -> Vec<&SemanticNode> {
        let mut result = vec![self];
        for child in &self.children {
            result.extend(child.iter_depth_first());
        }
        result
    }

    /// Collect all leaf nodes (no children)
    pub fn iter_leaves(&self) -> Vec<&SemanticNode> {
        if self.children.is_empty() {
            return vec![self];
        }
        self.children.iter().flat_map(|c| c.iter_leaves()).collect()
    }

    /// How many extra tokens we'd spend by unrolling this node's children (or raw text for leaves)
    pub fn unfold_cost(&self) -> u32 {
        if !self.children.is_empty() {
            self.subtree_token_count.saturating_sub(self.token_count)
        } else {
            self.raw_text_tokens
        }
    }

    /// Check if this node has children to unfold or raw text to reveal
    pub fn is_foldable(&self) -> bool {
        !self.children.is_empty() || self.raw_text.is_some()
    }
}

impl SemanticTree {
    /// Find a node anywhere in the tree by ID
    pub fn find_node(&self, id: &str) -> Option<&SemanticNode> {
        for root in &self.root_nodes {
            if let Some(found) = root.find(id) {
                return Some(found);
            }
        }
        None
    }

    /// Find a mutable node anywhere in the tree by ID
    pub fn find_node_mut(&mut self, id: &str) -> Option<&mut SemanticNode> {
        for root in &mut self.root_nodes {
            if let Some(found) = root.find_mut(id) {
                return Some(found);
            }
        }
        None
    }

    /// Collect all nodes in the tree depth-first
    pub fn all_nodes(&self) -> Vec<&SemanticNode> {
        self.root_nodes.iter().flat_map(|r| r.iter_depth_first()).collect()
    }

    /// Find the ancestor path from root to a given node ID (exclusive of the node itself).
    /// Returns the IDs of each ancestor in order from root → parent.
    /// Returns None if the node is not found.
    pub fn ancestor_path(&self, target_id: &str) -> Option<Vec<String>> {
        fn find_path(node: &SemanticNode, target: &str, path: &mut Vec<String>) -> bool {
            if node.id == target {
                return true;
            }
            path.push(node.id.clone());
            for child in &node.children {
                if find_path(child, target, path) {
                    return true;
                }
            }
            path.pop();
            false
        }

        for root in &self.root_nodes {
            let mut path = vec![];
            if find_path(root, target_id, &mut path) {
                return Some(path);
            }
        }
        None
    }

    /// Compression ratio: compressed / full
    pub fn compression_ratio(&self) -> f32 {
        if self.full_token_count == 0 {
            return 0.0;
        }
        self.compressed_token_count as f32 / self.full_token_count as f32
    }
}
