use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Action {
    Click {
        selector: String,
        description: String,
    },
    Fill {
        selector: String,
        field_type: FieldType,
        description: String,
    },
    Select {
        selector: String,
        options: Vec<String>,
        description: String,
    },
    Toggle {
        selector: String,
        description: String,
        current_state: bool,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FieldType {
    Text,
    Password,
    Email,
    Number,
    Search,
    Url,
    Textarea,
}

impl Action {
    pub fn selector(&self) -> &str {
        match self {
            Action::Click { selector, .. } => selector,
            Action::Fill { selector, .. } => selector,
            Action::Select { selector, .. } => selector,
            Action::Toggle { selector, .. } => selector,
        }
    }

    pub fn description(&self) -> &str {
        match self {
            Action::Click { description, .. } => description,
            Action::Fill { description, .. } => description,
            Action::Select { description, .. } => description,
            Action::Toggle { description, .. } => description,
        }
    }
}
