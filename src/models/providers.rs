//! Internal provider data models
//!
//! These types represent the normalized internal format used for communication
//! with all backend providers (OpenAI, Gemini, Ollama).

use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

use super::anthropic::Usage;

// ============================================================================
// Request Types
// ============================================================================

/// Normalized request format for all providers
#[derive(Debug, Clone)]
pub struct ProviderRequest {
    /// Model identifier
    pub model: String,

    /// Messages in the conversation
    pub messages: Vec<ProviderMessage>,

    /// Maximum tokens to generate
    pub max_tokens: u32,

    /// Sampling temperature (0.0-1.0)
    pub temperature: f32,

    /// Available tools (optional)
    pub tools: Option<Vec<ProviderTool>>,

    /// Tool choice directive
    pub tool_choice: Option<ProviderToolChoice>,

    /// Stop sequences
    pub stop: Option<Vec<String>>,
}

/// Message in provider format
#[derive(Debug, Clone)]
pub struct ProviderMessage {
    /// Message role ("system", "user", "assistant")
    pub role: String,

    /// Message content
    pub content: ProviderContent,
}

/// Message content in provider format
#[derive(Debug, Clone)]
pub enum ProviderContent {
    /// Plain text content
    Text(String),

    /// Structured content with multiple parts
    Parts(Vec<ProviderContentPart>),
}

/// Content part (for multimodal messages)
#[derive(Debug, Clone)]
pub enum ProviderContentPart {
    /// Text part
    Text { text: String },

    /// Image part (base64 or URL)
    Image {
        image_type: String, // "base64" or "url"
        source: String,     // base64 data or URL
        media_type: Option<String>,
    },
}

/// Tool definition in provider format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderTool {
    /// Tool name
    pub name: String,

    /// Tool description
    pub description: String,

    /// Parameters schema (JSON Schema)
    pub parameters: JsonValue,
}

/// Tool choice in provider format
#[derive(Debug, Clone)]
pub enum ProviderToolChoice {
    /// Automatic tool selection
    Auto,

    /// Required to use any tool
    Required,

    /// Specific tool must be used
    Specific(String),
}

// ============================================================================
// Response Types
// ============================================================================

/// Normalized response from providers
#[derive(Debug, Clone)]
pub struct ProviderResponse {
    /// Response ID
    pub id: String,

    /// Content blocks in the response
    pub content: Vec<ProviderContentBlock>,

    /// Why generation stopped
    pub stop_reason: StopReason,

    /// Token usage
    pub usage: Usage,
}

/// Content block in provider response
#[derive(Debug, Clone)]
pub enum ProviderContentBlock {
    /// Text content
    Text(String),

    /// Tool call from the assistant
    ToolCall {
        id: String,
        name: String,
        arguments: String, // JSON string
    },
}

/// Reason for stopping generation
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StopReason {
    /// Natural stop (completed response)
    EndTurn,

    /// Reached max tokens limit
    MaxTokens,

    /// Model wants to use a tool
    ToolUse,

    /// Hit a stop sequence
    StopSequence,

    /// Unknown/other reason
    Other(String),
}

impl StopReason {
    /// Convert to Anthropic stop_reason string
    pub fn to_anthropic(&self) -> String {
        match self {
            StopReason::EndTurn => "end_turn".to_string(),
            StopReason::MaxTokens => "max_tokens".to_string(),
            StopReason::ToolUse => "tool_use".to_string(),
            StopReason::StopSequence => "stop_sequence".to_string(),
            StopReason::Other(s) => s.clone(),
        }
    }

    /// Convert from OpenAI finish_reason
    pub fn from_openai(reason: &str) -> Self {
        match reason {
            "stop" => StopReason::EndTurn,
            "length" => StopReason::MaxTokens,
            "tool_calls" | "function_call" => StopReason::ToolUse,
            other => StopReason::Other(other.to_string()),
        }
    }
}

// ============================================================================
// Streaming Types
// ============================================================================

/// Stream chunk from providers
#[derive(Debug, Clone)]
pub enum StreamChunk {
    /// Text content delta
    TextDelta(String),

    /// Tool call started
    ToolCallStart { id: String, name: String },

    /// Tool call arguments delta (incremental JSON)
    ToolCallDelta { index: usize, json_delta: String },

    /// Stream finished
    Done {
        stop_reason: StopReason,
        usage: Usage,
    },

    /// Error during streaming
    Error { message: String },
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_request_creation() {
        let req = ProviderRequest {
            model: "gpt-4o".to_string(),
            messages: vec![ProviderMessage {
                role: "user".to_string(),
                content: ProviderContent::Text("Hello".to_string()),
            }],
            max_tokens: 100,
            temperature: 0.7,
            tools: None,
            tool_choice: None,
            stop: None,
        };

        assert_eq!(req.model, "gpt-4o");
        assert_eq!(req.messages.len(), 1);
        assert_eq!(req.max_tokens, 100);
    }

    #[test]
    fn test_provider_content_text() {
        let content = ProviderContent::Text("Test message".to_string());

        match content {
            ProviderContent::Text(text) => assert_eq!(text, "Test message"),
            _ => panic!("Expected Text variant"),
        }
    }

    #[test]
    fn test_provider_content_parts() {
        let content = ProviderContent::Parts(vec![
            ProviderContentPart::Text {
                text: "Hello".to_string(),
            },
            ProviderContentPart::Text {
                text: "World".to_string(),
            },
        ]);

        match content {
            ProviderContent::Parts(parts) => assert_eq!(parts.len(), 2),
            _ => panic!("Expected Parts variant"),
        }
    }

    #[test]
    fn test_provider_response() {
        let resp = ProviderResponse {
            id: "resp_123".to_string(),
            content: vec![ProviderContentBlock::Text("Hello!".to_string())],
            stop_reason: StopReason::EndTurn,
            usage: Usage {
                input_tokens: 10,
                output_tokens: 5,
            },
        };

        assert_eq!(resp.id, "resp_123");
        assert_eq!(resp.content.len(), 1);
        assert_eq!(resp.stop_reason, StopReason::EndTurn);
    }

    #[test]
    fn test_stop_reason_to_anthropic() {
        assert_eq!(StopReason::EndTurn.to_anthropic(), "end_turn");
        assert_eq!(StopReason::MaxTokens.to_anthropic(), "max_tokens");
        assert_eq!(StopReason::ToolUse.to_anthropic(), "tool_use");
        assert_eq!(StopReason::StopSequence.to_anthropic(), "stop_sequence");
    }

    #[test]
    fn test_stop_reason_from_openai() {
        assert_eq!(StopReason::from_openai("stop"), StopReason::EndTurn);
        assert_eq!(StopReason::from_openai("length"), StopReason::MaxTokens);
        assert_eq!(StopReason::from_openai("tool_calls"), StopReason::ToolUse);
        assert_eq!(
            StopReason::from_openai("function_call"),
            StopReason::ToolUse
        );

        match StopReason::from_openai("custom") {
            StopReason::Other(s) => assert_eq!(s, "custom"),
            _ => panic!("Expected Other variant"),
        }
    }

    #[test]
    fn test_stream_chunk_variants() {
        let text_delta = StreamChunk::TextDelta("Hello".to_string());
        match text_delta {
            StreamChunk::TextDelta(text) => assert_eq!(text, "Hello"),
            _ => panic!("Expected TextDelta"),
        }

        let tool_start = StreamChunk::ToolCallStart {
            id: "call_123".to_string(),
            name: "calculator".to_string(),
        };
        match tool_start {
            StreamChunk::ToolCallStart { id, name } => {
                assert_eq!(id, "call_123");
                assert_eq!(name, "calculator");
            }
            _ => panic!("Expected ToolCallStart"),
        }

        let done = StreamChunk::Done {
            stop_reason: StopReason::EndTurn,
            usage: Usage {
                input_tokens: 10,
                output_tokens: 5,
            },
        };
        match done {
            StreamChunk::Done { stop_reason, usage } => {
                assert_eq!(stop_reason, StopReason::EndTurn);
                assert_eq!(usage.input_tokens, 10);
            }
            _ => panic!("Expected Done"),
        }
    }

    #[test]
    fn test_provider_tool_creation() {
        let tool = ProviderTool {
            name: "calculator".to_string(),
            description: "Performs calculations".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "expression": {"type": "string"}
                }
            }),
        };

        assert_eq!(tool.name, "calculator");
        assert_eq!(tool.description, "Performs calculations");
    }

    #[test]
    fn test_provider_tool_choice() {
        let auto = ProviderToolChoice::Auto;
        match auto {
            ProviderToolChoice::Auto => {}
            _ => panic!("Expected Auto"),
        }

        let required = ProviderToolChoice::Required;
        match required {
            ProviderToolChoice::Required => {}
            _ => panic!("Expected Required"),
        }

        let specific = ProviderToolChoice::Specific("calculator".to_string());
        match specific {
            ProviderToolChoice::Specific(name) => assert_eq!(name, "calculator"),
            _ => panic!("Expected Specific"),
        }
    }

    #[test]
    fn test_provider_content_block_text() {
        let block = ProviderContentBlock::Text("Hello".to_string());
        match block {
            ProviderContentBlock::Text(text) => assert_eq!(text, "Hello"),
            _ => panic!("Expected Text"),
        }
    }

    #[test]
    fn test_provider_content_block_tool_call() {
        let block = ProviderContentBlock::ToolCall {
            id: "call_123".to_string(),
            name: "calculator".to_string(),
            arguments: r#"{"expression":"2+2"}"#.to_string(),
        };

        match block {
            ProviderContentBlock::ToolCall { id, name, arguments } => {
                assert_eq!(id, "call_123");
                assert_eq!(name, "calculator");
                assert!(arguments.contains("2+2"));
            }
            _ => panic!("Expected ToolCall"),
        }
    }

    #[test]
    fn test_provider_message_with_parts() {
        let msg = ProviderMessage {
            role: "user".to_string(),
            content: ProviderContent::Parts(vec![
                ProviderContentPart::Text {
                    text: "What's in this image?".to_string(),
                },
                ProviderContentPart::Image {
                    image_type: "base64".to_string(),
                    source: "base64data".to_string(),
                    media_type: Some("image/png".to_string()),
                },
            ]),
        };

        assert_eq!(msg.role, "user");
        match msg.content {
            ProviderContent::Parts(parts) => assert_eq!(parts.len(), 2),
            _ => panic!("Expected Parts"),
        }
    }
}
