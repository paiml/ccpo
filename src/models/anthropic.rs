//! Anthropic API data models
//!
//! This module defines the request and response types that match the Anthropic Claude API format.
//! Reference: https://docs.anthropic.com/en/api/messages

use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

// ============================================================================
// Request Types
// ============================================================================

/// Request to create a message
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct MessagesRequest {
    /// The model to use (e.g., "claude-3-sonnet-20240229", "haiku", "sonnet")
    pub model: String,

    /// Maximum tokens to generate
    pub max_tokens: u32,

    /// List of messages in the conversation
    pub messages: Vec<Message>,

    /// System prompt (optional, can be string or array of content blocks)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<SystemPrompt>,

    /// Sampling temperature (0-1, default 1.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,

    /// Whether to stream the response
    #[serde(default)]
    pub stream: bool,

    /// Available tools for the model to use
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,

    /// How the model should use tools
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,

    /// Additional stop sequences
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
}

/// System prompt (can be string or array of text blocks)
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum SystemPrompt {
    /// Simple string system prompt
    Text(String),
    /// Array of content blocks
    Blocks(Vec<SystemBlock>),
}

impl SystemPrompt {
    /// Convert to a single string representation
    pub fn to_string(&self) -> String {
        match self {
            SystemPrompt::Text(s) => s.clone(),
            SystemPrompt::Blocks(blocks) => blocks
                .iter()
                .filter_map(|b| match b {
                    SystemBlock::Text { text, .. } => Some(text.as_str()),
                })
                .collect::<Vec<_>>()
                .join("\n"),
        }
    }
}

/// System prompt content block
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type")]
pub enum SystemBlock {
    #[serde(rename = "text")]
    Text {
        text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
}

/// Cache control for prompt caching
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CacheControl {
    #[serde(rename = "type")]
    pub cache_type: String,
}

/// A message in the conversation
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Message {
    /// Role: "user" or "assistant"
    pub role: String,

    /// Message content (can be string or array of content blocks)
    pub content: Content,
}

/// Message content (string or structured blocks)
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum Content {
    /// Simple text content
    Text(String),
    /// Structured content blocks
    Blocks(Vec<ContentBlock>),
}

/// Content block in a message
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type")]
pub enum ContentBlock {
    /// Text content
    #[serde(rename = "text")]
    Text { text: String },

    /// Image content
    #[serde(rename = "image")]
    Image { source: ImageSource },

    /// Tool use request from assistant
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: JsonValue,
    },

    /// Tool result from user
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        #[serde(flatten)]
        content: ToolResultContent,
    },
}

impl ContentBlock {
    /// Get text content if this is a Text block
    pub fn as_text(&self) -> Option<&str> {
        match self {
            ContentBlock::Text { text } => Some(text),
            _ => None,
        }
    }

    /// Check if this is a tool result block
    pub fn is_tool_result(&self) -> bool {
        matches!(self, ContentBlock::ToolResult { .. })
    }
}

/// Image source
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ImageSource {
    #[serde(rename = "type")]
    pub source_type: String, // "base64" or "url"

    #[serde(skip_serializing_if = "Option::is_none")]
    pub media_type: Option<String>, // "image/jpeg", "image/png", etc.

    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<String>, // Base64 encoded image

    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>, // Image URL
}

/// Tool result content (can be string or array of blocks)
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum ToolResultContent {
    /// Simple text result
    Text { content: String },
    /// Structured content blocks
    Blocks { content: Vec<ContentBlock> },
}

impl ToolResultContent {
    /// Extract text from tool result content
    pub fn to_text(&self) -> String {
        match self {
            ToolResultContent::Text { content } => content.clone(),
            ToolResultContent::Blocks { content } => content
                .iter()
                .filter_map(|b| b.as_text())
                .collect::<Vec<_>>()
                .join("\n"),
        }
    }
}

/// Tool definition
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Tool {
    /// Tool name
    pub name: String,

    /// Tool description
    pub description: String,

    /// Input schema (JSON Schema)
    pub input_schema: JsonValue,
}

/// Tool choice directive
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum ToolChoice {
    /// Automatic tool selection
    Auto { r#type: String }, // type: "auto"

    /// Any tool must be used
    Any { r#type: String }, // type: "any"

    /// Specific tool must be used
    Tool {
        r#type: String, // type: "tool"
        name: String,
    },
}

// ============================================================================
// Response Types
// ============================================================================

/// Response from creating a message
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct MessagesResponse {
    /// Unique message ID
    pub id: String,

    /// Model that generated the response
    pub model: String,

    /// Role (always "assistant")
    pub role: String,

    /// Response content
    pub content: Vec<ContentBlock>,

    /// Type (always "message")
    #[serde(rename = "type")]
    pub msg_type: String,

    /// Why the model stopped generating
    pub stop_reason: Option<String>,

    /// Stop sequence that triggered the stop (if any)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequence: Option<String>,

    /// Token usage statistics
    pub usage: Usage,
}

/// Token usage statistics
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct Usage {
    /// Number of input tokens
    pub input_tokens: u32,

    /// Number of output tokens
    pub output_tokens: u32,
}

/// Token count request
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TokenCountRequest {
    /// Model to count tokens for
    pub model: String,

    /// Messages to count
    pub messages: Vec<Message>,

    /// System prompt
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<SystemPrompt>,

    /// Tools (affects token count)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
}

/// Token count response
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TokenCountResponse {
    /// Number of input tokens
    pub input_tokens: u32,
}

// ============================================================================
// Streaming Types
// ============================================================================

/// Server-Sent Event types for streaming responses
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
pub enum StreamEvent {
    /// Stream started
    #[serde(rename = "message_start")]
    MessageStart { message: MessageStart },

    /// Content block started
    #[serde(rename = "content_block_start")]
    ContentBlockStart {
        index: usize,
        content_block: ContentBlock,
    },

    /// Content block delta (incremental update)
    #[serde(rename = "content_block_delta")]
    ContentBlockDelta { index: usize, delta: Delta },

    /// Content block stopped
    #[serde(rename = "content_block_stop")]
    ContentBlockStop { index: usize },

    /// Message delta (final metadata)
    #[serde(rename = "message_delta")]
    MessageDelta {
        delta: MessageDeltaData,
        usage: Usage,
    },

    /// Stream ended
    #[serde(rename = "message_stop")]
    MessageStop,

    /// Keepalive ping
    #[serde(rename = "ping")]
    Ping,

    /// Error during streaming
    #[serde(rename = "error")]
    Error { error: ErrorData },
}

/// Message metadata at stream start
#[derive(Debug, Clone, Serialize)]
pub struct MessageStart {
    pub id: String,
    #[serde(rename = "type")]
    pub msg_type: String,
    pub role: String,
    pub model: String,
    pub content: Vec<ContentBlock>,
    pub stop_reason: Option<String>,
    pub usage: Usage,
}

/// Delta (incremental content update)
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
pub enum Delta {
    /// Text delta
    #[serde(rename = "text_delta")]
    TextDelta { text: String },

    /// Tool input JSON delta
    #[serde(rename = "input_json_delta")]
    InputJsonDelta { partial_json: String },
}

/// Message delta data (final metadata)
#[derive(Debug, Clone, Serialize)]
pub struct MessageDeltaData {
    pub stop_reason: String,
    pub stop_sequence: Option<String>,
}

/// Error data
#[derive(Debug, Clone, Serialize)]
pub struct ErrorData {
    #[serde(rename = "type")]
    pub error_type: String,
    pub message: String,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_messages_request_simple() {
        let json = json!({
            "model": "claude-3-sonnet-20240229",
            "max_tokens": 1024,
            "messages": [
                {
                    "role": "user",
                    "content": "Hello"
                }
            ]
        });

        let req: MessagesRequest = serde_json::from_value(json.clone()).unwrap();
        assert_eq!(req.model, "claude-3-sonnet-20240229");
        assert_eq!(req.max_tokens, 1024);
        assert_eq!(req.messages.len(), 1);
        assert_eq!(req.stream, false);

        // Round-trip test
        let serialized = serde_json::to_value(&req).unwrap();
        let deserialized: MessagesRequest = serde_json::from_value(serialized).unwrap();
        assert_eq!(deserialized.model, req.model);
    }

    #[test]
    fn test_content_text_variant() {
        let json = json!("Hello, world!");
        let content: Content = serde_json::from_value(json).unwrap();

        match content {
            Content::Text(text) => assert_eq!(text, "Hello, world!"),
            _ => panic!("Expected Text variant"),
        }
    }

    #[test]
    fn test_content_blocks_variant() {
        let json = json!([
            {"type": "text", "text": "Hello"},
            {"type": "text", "text": "World"}
        ]);

        let content: Content = serde_json::from_value(json).unwrap();

        match content {
            Content::Blocks(blocks) => {
                assert_eq!(blocks.len(), 2);
                assert_eq!(blocks[0].as_text(), Some("Hello"));
                assert_eq!(blocks[1].as_text(), Some("World"));
            }
            _ => panic!("Expected Blocks variant"),
        }
    }

    #[test]
    fn test_content_block_text() {
        let json = json!({"type": "text", "text": "Test message"});
        let block: ContentBlock = serde_json::from_value(json).unwrap();

        assert_eq!(block.as_text(), Some("Test message"));
    }

    #[test]
    fn test_content_block_tool_use() {
        let json = json!({
            "type": "tool_use",
            "id": "toolu_123",
            "name": "calculator",
            "input": {"expression": "2+2"}
        });

        let block: ContentBlock = serde_json::from_value(json.clone()).unwrap();

        match &block {
            ContentBlock::ToolUse { id, name, input } => {
                assert_eq!(id, "toolu_123");
                assert_eq!(name, "calculator");
                assert_eq!(input["expression"], "2+2");
            }
            _ => panic!("Expected ToolUse variant"),
        }

        // Round-trip
        let serialized = serde_json::to_value(&block).unwrap();
        assert_eq!(serialized["type"], "tool_use");
    }

    #[test]
    fn test_content_block_tool_result() {
        let json = json!({
            "type": "tool_result",
            "tool_use_id": "toolu_123",
            "content": "4"
        });

        let block: ContentBlock = serde_json::from_value(json).unwrap();

        match block {
            ContentBlock::ToolResult { tool_use_id, content } => {
                assert_eq!(tool_use_id, "toolu_123");
                assert_eq!(content.to_text(), "4");
            }
            _ => panic!("Expected ToolResult variant"),
        }
    }

    #[test]
    fn test_messages_response() {
        let json = json!({
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-sonnet-20240229",
            "content": [
                {"type": "text", "text": "Hello!"}
            ],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5
            }
        });

        let resp: MessagesResponse = serde_json::from_value(json).unwrap();
        assert_eq!(resp.id, "msg_123");
        assert_eq!(resp.role, "assistant");
        assert_eq!(resp.content.len(), 1);
        assert_eq!(resp.stop_reason, Some("end_turn".to_string()));
        assert_eq!(resp.usage.input_tokens, 10);
        assert_eq!(resp.usage.output_tokens, 5);
    }

    #[test]
    fn test_system_prompt_text() {
        let json = json!("You are a helpful assistant");
        let system: SystemPrompt = serde_json::from_value(json).unwrap();

        assert_eq!(system.to_string(), "You are a helpful assistant");
    }

    #[test]
    fn test_system_prompt_blocks() {
        let json = json!([
            {"type": "text", "text": "You are helpful"},
            {"type": "text", "text": "Be concise"}
        ]);

        let system: SystemPrompt = serde_json::from_value(json).unwrap();
        assert_eq!(system.to_string(), "You are helpful\nBe concise");
    }

    #[test]
    fn test_tool_definition() {
        let json = json!({
            "name": "calculator",
            "description": "Performs calculations",
            "input_schema": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string"}
                },
                "required": ["expression"]
            }
        });

        let tool: Tool = serde_json::from_value(json.clone()).unwrap();
        assert_eq!(tool.name, "calculator");

        // Round-trip
        let serialized = serde_json::to_value(&tool).unwrap();
        let deserialized: Tool = serde_json::from_value(serialized).unwrap();
        assert_eq!(deserialized.name, tool.name);
    }

    #[test]
    fn test_empty_messages_array() {
        let json = json!({
            "model": "haiku",
            "max_tokens": 100,
            "messages": []
        });

        let req: MessagesRequest = serde_json::from_value(json).unwrap();
        assert_eq!(req.messages.len(), 0);
    }

    #[test]
    fn test_null_optional_fields() {
        let json = json!({
            "model": "sonnet",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Hi"}],
            "system": null,
            "tools": null
        });

        let req: MessagesRequest = serde_json::from_value(json).unwrap();
        assert!(req.system.is_none());
        assert!(req.tools.is_none());
    }

    #[test]
    fn test_streaming_request() {
        let json = json!({
            "model": "haiku",
            "max_tokens": 100,
            "stream": true,
            "messages": [{"role": "user", "content": "Count to 5"}]
        });

        let req: MessagesRequest = serde_json::from_value(json).unwrap();
        assert_eq!(req.stream, true);
    }

    #[test]
    fn test_usage_default() {
        let usage = Usage::default();
        assert_eq!(usage.input_tokens, 0);
        assert_eq!(usage.output_tokens, 0);
    }

    #[test]
    fn test_tool_result_content_variants() {
        // Text variant
        let json = json!({"content": "Result text"});
        let content: ToolResultContent = serde_json::from_value(json).unwrap();
        assert_eq!(content.to_text(), "Result text");

        // Blocks variant
        let json = json!({
            "content": [
                {"type": "text", "text": "Line 1"},
                {"type": "text", "text": "Line 2"}
            ]
        });
        let content: ToolResultContent = serde_json::from_value(json).unwrap();
        assert_eq!(content.to_text(), "Line 1\nLine 2");
    }

    #[test]
    fn test_malformed_json_fails() {
        let json = json!({
            "model": "test",
            // missing required max_tokens
            "messages": []
        });

        let result: Result<MessagesRequest, _> = serde_json::from_value(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_content_block_is_tool_result() {
        let tool_result = ContentBlock::ToolResult {
            tool_use_id: "test".to_string(),
            content: ToolResultContent::Text {
                content: "result".to_string(),
            },
        };
        assert!(tool_result.is_tool_result());

        let text_block = ContentBlock::Text {
            text: "test".to_string(),
        };
        assert!(!text_block.is_tool_result());
    }

    #[test]
    fn test_token_count_request() {
        let json = json!({
            "model": "haiku",
            "messages": [
                {"role": "user", "content": "Hello"}
            ]
        });

        let req: TokenCountRequest = serde_json::from_value(json).unwrap();
        assert_eq!(req.model, "haiku");
        assert_eq!(req.messages.len(), 1);
    }
}
