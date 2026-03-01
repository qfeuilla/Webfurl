use sha2::{Digest, Sha256};

/// Compute a structural hash for a DOM node based on its tag, classes, and children's hashes.
/// This ignores text content, so pages with same structure but different data produce the same hash.
pub fn structural_hash(
    tag: &str,
    classes: &[&str],
    children_hashes: &[&str],
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(tag.as_bytes());
    hasher.update(b"|");

    let mut sorted_classes = classes.to_vec();
    sorted_classes.sort();
    for class in &sorted_classes {
        hasher.update(class.as_bytes());
        hasher.update(b",");
    }
    hasher.update(b"|");

    for child_hash in children_hashes {
        hasher.update(child_hash.as_bytes());
        hasher.update(b";");
    }

    hex::encode(hasher.finalize())
}

/// Compute a content hash for cache invalidation of dynamic slots.
/// This hashes the actual text content of a node.
pub fn content_hash(text: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    hex::encode(hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_structure_same_hash() {
        let h1 = structural_hash("div", &["container", "main"], &[]);
        let h2 = structural_hash("div", &["main", "container"], &[]);
        assert_eq!(h1, h2, "class order shouldn't matter");
    }

    #[test]
    fn different_tag_different_hash() {
        let h1 = structural_hash("div", &["container"], &[]);
        let h2 = structural_hash("section", &["container"], &[]);
        assert_ne!(h1, h2);
    }

    #[test]
    fn children_affect_hash() {
        let child_hash = structural_hash("span", &[], &[]);
        let h1 = structural_hash("div", &[], &[&child_hash]);
        let h2 = structural_hash("div", &[], &[]);
        assert_ne!(h1, h2);
    }
}
