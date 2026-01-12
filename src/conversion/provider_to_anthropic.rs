//! Convert provider responses back to Anthropic API format
//!
//! This module implements the conversion from our internal provider format
//! back to Anthropic's Claude API format. It handles:
//!
//! - Content extraction from provider responses
//! - Stop reason mapping (stop → end_turn, length → max_tokens, etc.)
//! - Usage token extraction
//! - Fallback handling (always include at least one content block)

use crate::error::ProxyResult;
use crate::models::anthropic::{ContentBlock, MessagesResponse};
use crate::models::providers::{ProviderContentBlock, ProviderResponse};
use serde_json;

/// Convert a provider response to Anthropic format
///
/// This function:
/// 1. Extracts content blocks from the provider response
/// 2. Maps the stop reason to Anthropic format
/// 3. Extracts usage information
/// 4. Ensures at least one content block is present (adds empty text if needed)
///
/// # Arguments
///
/// * `response` - The provider response
/// * `original_model` - The original model name from the request (for the response)
///
/// # Errors
///
/// Returns `ConversionError` if the response cannot be converted
pub fn convert_provider_to_anthropic(
    response: &ProviderResponse,
    original_model: &str,
) -> ProxyResult<MessagesResponse> {
    // Extract content blocks
    let mut content = Vec::new();

    for block in &response.content {
        match block {
            ProviderContentBlock::Text(text) => {
                content.push(ContentBlock::Text {
                    text: text.clone(),
                });
            }
            ProviderContentBlock::ToolCall { id, name, arguments } => {
                // Parse arguments from JSON string to Value
                let input = serde_json::from_str(arguments).unwrap_or_else(|_| {
                    serde_json::json!({})
                });

                content.push(ContentBlock::ToolUse {
                    id: id.clone(),
                    name: name.clone(),
                    input,
                });
            }
        }
    }

    // Fallback: if no content blocks, add an empty text block
    if content.is_empty() {
        content.push(ContentBlock::Text {
            text: String::new(),
        });
    }

    // Map stop reason
    let stop_reason = Some(response.stop_reason.to_anthropic());

    Ok(MessagesResponse {
        id: response.id.clone(),
        model: original_model.to_string(),
        role: "assistant".to_string(),
        content,
        msg_type: "message".to_string(),
        stop_reason,
        stop_sequence: None,
        usage: response.usage.clone(),
    })
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::anthropic::Usage;
    use crate::models::providers::StopReason;

    #[test]
    fn test_convert_simple_text_response() {
        let provider_response = ProviderResponse {
            id: "resp_123".to_string(),
            content: vec![ProviderContentBlock::Text(
                "Hello, world!".to_string(),
            )],
            stop_reason: StopReason::EndTurn,
            usage: Usage {
                input_tokens: 10,
                output_tokens: 5,
            },
        };

        let result = convert_provider_to_anthropic(&provider_response, "gpt-4").unwrap();

        assert_eq!(result.model, "gpt-4");
        assert_eq!(result.role, "assistant");
        assert_eq!(result.content.len(), 1);
        assert_eq!(result.stop_reason, Some("end_turn".to_string()));
        assert_eq!(result.usage.input_tokens, 10);
        assert_eq!(result.usage.output_tokens, 5);

        if let ContentBlock::Text { text } = &result.content[0] {
            assert_eq!(text, "Hello, world!");
        } else {
            panic!("Expected text content block");
        }
    }

    #[test]
    fn test_convert_with_tool_call() {
        let provider_response = ProviderResponse {
            id: "resp_456".to_string(),
            content: vec![
                ProviderContentBlock::Text("Let me calculate that".to_string()),
                ProviderContentBlock::ToolCall {
                    id: "call_123".to_string(),
                    name: "calculator".to_string(),
                    arguments: r#"{"expression":"2+2"}"#.to_string(),
                },
            ],
            stop_reason: StopReason::ToolUse,
            usage: Usage {
                input_tokens: 20,
                output_tokens: 15,
            },
        };

        let result = convert_provider_to_anthropic(&provider_response, "gpt-4").unwrap();

        assert_eq!(result.content.len(), 2);
        assert_eq!(result.stop_reason, Some("tool_use".to_string()));

        // Check text block
        if let ContentBlock::Text { text } = &result.content[0] {
            assert_eq!(text, "Let me calculate that");
        } else {
            panic!("Expected text content block");
        }

        // Check tool use block
        if let ContentBlock::ToolUse { id, name, input } = &result.content[1] {
            assert_eq!(id, "call_123");
            assert_eq!(name, "calculator");
            assert_eq!(input["expression"], "2+2");
        } else {
            panic!("Expected tool use content block");
        }
    }

    #[test]
    fn test_convert_stop_reason_max_tokens() {
        let provider_response = ProviderResponse {
            id: "resp_789".to_string(),
            content: vec![ProviderContentBlock::Text("Incomplete...".to_string())],
            stop_reason: StopReason::MaxTokens,
            usage: Usage {
                input_tokens: 10,
                output_tokens: 100,
            },
        };

        let result = convert_provider_to_anthropic(&provider_response, "gpt-4").unwrap();

        assert_eq!(result.stop_reason, Some("max_tokens".to_string()));
    }

    #[test]
    fn test_convert_stop_reason_stop_sequence() {
        let provider_response = ProviderResponse {
            id: "resp_abc".to_string(),
            content: vec![ProviderContentBlock::Text("Done".to_string())],
            stop_reason: StopReason::StopSequence,
            usage: Usage {
                input_tokens: 10,
                output_tokens: 5,
            },
        };

        let result = convert_provider_to_anthropic(&provider_response, "gpt-4").unwrap();

        assert_eq!(result.stop_reason, Some("stop_sequence".to_string()));
    }

    #[test]
    fn test_convert_empty_content_adds_fallback() {
        let provider_response = ProviderResponse {
            id: "resp_empty".to_string(),
            content: vec![],
            stop_reason: StopReason::EndTurn,
            usage: Usage {
                input_tokens: 10,
                output_tokens: 0,
            },
        };

        let result = convert_provider_to_anthropic(&provider_response, "gpt-4").unwrap();

        assert_eq!(result.content.len(), 1);
        if let ContentBlock::Text { text } = &result.content[0] {
            assert_eq!(text, "");
        } else {
            panic!("Expected empty text content block");
        }
    }

    #[test]
    fn test_convert_preserves_id() {
        let provider_response = ProviderResponse {
            id: "unique_id_123".to_string(),
            content: vec![ProviderContentBlock::Text("Hello".to_string())],
            stop_reason: StopReason::EndTurn,
            usage: Usage {
                input_tokens: 10,
                output_tokens: 5,
            },
        };

        let result = convert_provider_to_anthropic(&provider_response, "gpt-4").unwrap();

        assert_eq!(result.id, "unique_id_123");
    }

    #[test]
    fn test_convert_preserves_model_name() {
        let provider_response = ProviderResponse {
            id: "resp_model".to_string(),
            content: vec![ProviderContentBlock::Text("Hello".to_string())],
            stop_reason: StopReason::EndTurn,
            usage: Usage {
                input_tokens: 10,
                output_tokens: 5,
            },
        };

        let result = convert_provider_to_anthropic(&provider_response, "claude-3-haiku-20240307").unwrap();

        assert_eq!(result.model, "claude-3-haiku-20240307");
    }

    #[test]
    fn test_convert_tool_call_with_invalid_json() {
        let provider_response = ProviderResponse {
            id: "resp_invalid".to_string(),
            content: vec![ProviderContentBlock::ToolCall {
                id: "call_bad".to_string(),
                name: "calculator".to_string(),
                arguments: "not valid json".to_string(),
            }],
            stop_reason: StopReason::ToolUse,
            usage: Usage {
                input_tokens: 10,
                output_tokens: 5,
            },
        };

        let result = convert_provider_to_anthropic(&provider_response, "gpt-4").unwrap();

        // Should not panic, should use empty object as fallback
        if let ContentBlock::ToolUse { input, .. } = &result.content[0] {
            assert_eq!(input, &serde_json::json!({}));
        } else {
            panic!("Expected tool use content block");
        }
    }
}
