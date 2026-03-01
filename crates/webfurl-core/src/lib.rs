pub mod tree;
pub mod actions;
pub mod hasher;
pub mod embeddings;
pub mod pipeline;
pub mod cache;
pub mod unfold;
pub mod serialize;
pub mod debug;
pub mod vision;
pub mod error;

pub use tree::{SemanticNode, SemanticTree};
pub use actions::Action;
pub use error::WebfurlError;

pub type Result<T> = std::result::Result<T, WebfurlError>;
