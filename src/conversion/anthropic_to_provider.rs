//! Convert Anthropic API format to internal provider format
//!
//! This module implements the conversion from Anthropic's Claude API format
//! to our internal provider format. It handles:
//!
//! - System message extraction and formatting
//! - Content block processing (text, images, tool use, tool results)
//! - Tool result flattening for OpenAI compatibility
//! - Gemini schema cleaning (recursive removal of unsupported properties)
//! - Max tokens capping for providers with limits

use crate::error::ProxyResult;
use crate::models::anthropic::{
    ContentBlock, Message, MessagesRequest, SystemPrompt, Tool, ToolChoice,
    ToolResultContent,
};
use crate::models::providers::{
    ProviderContent, ProviderContentPart, ProviderMessage, ProviderRequest, ProviderTool,
    ProviderToolChoice,
};
use serde_json::Value;

/// Maximum tokens allowed by OpenAI and Gemini
const MAX_TOKENS_LIMIT: u32 = 16384;

/// Convert an Anthropic request to internal provider format
///
/// This function performs the following transformations:
/// 1. Extracts system messages and inserts them as the first message
/// 2. Processes all content blocks (text, images, tool use, tool results)
/// 3. Flattens tool results for OpenAI compatibility
/// 4. Converts tools to provider format
/// 5. Caps max_tokens if needed
///
/// # Arguments
///
/// * `request` - The Anthropic API request
/// * `provider` - The target provider ("openai", "gemini", "ollama")
///
/// # Errors
///
/// Returns `ConversionError` if the request cannot be converted
pub fn convert_anthropic_to_provider(
    request: &MessagesRequest,
    provider: &str,
) -> ProxyResult<ProviderRequest> {
    let mut messages = Vec::new();

    // Step 1: Handle system message
    if let Some(system) = &request.system {
        messages.push(create_system_message(system));
    }

    // Step 2: Convert all messages
    for message in &request.messages {
        let provider_message = convert_message(message, provider)?;
        messages.push(provider_message);
    }

    // Step 3: Convert tools if present
    let tools = request
        .tools
        .as_ref()
        .map(|tools| convert_tools(tools, provider))
        .transpose()?;

    // Step 4: Convert tool choice if present
    let tool_choice = request
        .tool_choice
        .as_ref()
        .map(|choice| convert_tool_choice(choice));

    // Step 5: Cap max_tokens for OpenAI and Gemini
    let max_tokens = if provider == "openai" || provider == "gemini" || provider == "google" {
        std::cmp::min(request.max_tokens, MAX_TOKENS_LIMIT)
    } else {
        request.max_tokens
    };

    Ok(ProviderRequest {
        model: request.model.clone(),
        messages,
        max_tokens,
        temperature: request.temperature.unwrap_or(1.0),
        tools,
        tool_choice,
        stop: None,
    })
}

/// Create a system message from the Anthropic system prompt
fn create_system_message(system: &SystemPrompt) -> ProviderMessage {
    let content = match system {
        SystemPrompt::Text(text) => text.clone(),
        SystemPrompt::Blocks(blocks) => {
            // Concatenate all text blocks
            blocks
                .iter()
                .filter_map(|block| match block {
                    crate::models::anthropic::SystemBlock::Text { text, .. } => Some(text.as_str()),
                })
                .collect::<Vec<_>>()
                .join("\n")
        }
    };

    ProviderMessage {
        role: "system".to_string(),
        content: ProviderContent::Text(content),
    }
}

/// Convert an Anthropic message to provider format
fn convert_message(message: &Message, _provider: &str) -> ProxyResult<ProviderMessage> {
    let content = match &message.content {
        crate::models::anthropic::Content::Text(text) => {
            ProviderContent::Text(text.clone())
        }
        crate::models::anthropic::Content::Blocks(blocks) => {
            let mut parts = Vec::new();

            for block in blocks {
                match block {
                    ContentBlock::Text { text } => {
                        parts.push(ProviderContentPart::Text {
                            text: text.clone(),
                        });
                    }
                    ContentBlock::Image { source } => {
                        // Convert image to provider format
                        parts.push(ProviderContentPart::Image {
                            image_type: source.source_type.clone(),
                            source: source.data.clone().unwrap_or_default(),
                            media_type: source.media_type.clone(),
                        });
                    }
                    ContentBlock::ToolUse { id, name, input } => {
                        // Note: Tool calls are not supported in the Parts format
                        // They would need special handling in the actual provider conversion
                        // For now, we represent them as text
                        let tool_text = format!(
                            "Tool call: {} (id: {}) with args: {}",
                            name,
                            id,
                            serde_json::to_string(input).unwrap_or_default()
                        );
                        parts.push(ProviderContentPart::Text { text: tool_text });
                    }
                    ContentBlock::ToolResult {
                        tool_use_id,
                        content,
                    } => {
                        // CRITICAL: Flatten tool results to text for OpenAI compatibility
                        let text = flatten_tool_result(tool_use_id, content);
                        parts.push(ProviderContentPart::Text { text });
                    }
                }
            }

            // If we have multiple parts or any non-text content, use Parts
            // Otherwise simplify to Text
            if parts.len() == 1 {
                if let ProviderContentPart::Text { text } = &parts[0] {
                    ProviderContent::Text(text.clone())
                } else {
                    ProviderContent::Parts(parts)
                }
            } else {
                ProviderContent::Parts(parts)
            }
        }
    };

    Ok(ProviderMessage {
        role: message.role.clone(),
        content,
    })
}

/// Flatten a tool result to text format for OpenAI compatibility
///
/// OpenAI requires tool results to be in text format in user messages.
/// This function formats them as: "Tool result for {id}:\n{content}\n"
fn flatten_tool_result(tool_use_id: &str, content: &ToolResultContent) -> String {
    let content_text = match content {
        ToolResultContent::Text { content } => content.clone(),
        ToolResultContent::Blocks { content: blocks } => blocks
            .iter()
            .filter_map(|b| match b {
                ContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n"),
    };

    format!("Tool result for {}:\n{}\n", tool_use_id, content_text)
}

/// Convert Anthropic tools to provider format
fn convert_tools(tools: &[Tool], provider: &str) -> ProxyResult<Vec<ProviderTool>> {
    tools
        .iter()
        .map(|tool| {
            let mut schema = tool.input_schema.clone();

            // Clean schema for Gemini
            if provider == "gemini" || provider == "google" {
                schema = clean_schema_for_gemini(schema)?;
            }

            Ok(ProviderTool {
                name: tool.name.clone(),
                description: tool.description.clone(),
                parameters: schema,
            })
        })
        .collect()
}

/// Clean JSON schema for Gemini compatibility
///
/// Gemini has strict schema requirements and doesn't support:
/// - `additionalProperties`
/// - `default` values
/// - String formats except `enum` and `date-time`
///
/// This function recursively removes these unsupported properties.
fn clean_schema_for_gemini(mut schema: Value) -> ProxyResult<Value> {
    if let Some(obj) = schema.as_object_mut() {
        // Remove unsupported properties at this level
        obj.remove("additionalProperties");
        obj.remove("default");

        // Clean string format if present
        if let Some(Value::String(format)) = obj.get("format") {
            if format != "enum" && format != "date-time" {
                obj.remove("format");
            }
        }

        // Recursively clean nested objects
        if let Some(Value::Object(properties)) = obj.get_mut("properties") {
            for (_key, value) in properties.iter_mut() {
                *value = clean_schema_for_gemini(value.clone())?;
            }
        }

        // Recursively clean items in arrays
        if let Some(items) = obj.get_mut("items") {
            *items = clean_schema_for_gemini(items.clone())?;
        }

        // Recursively clean oneOf/anyOf/allOf
        for key in &["oneOf", "anyOf", "allOf"] {
            if let Some(Value::Array(variants)) = obj.get_mut(*key) {
                for variant in variants.iter_mut() {
                    *variant = clean_schema_for_gemini(variant.clone())?;
                }
            }
        }
    }

    Ok(schema)
}

/// Convert Anthropic tool choice to provider format
fn convert_tool_choice(choice: &ToolChoice) -> ProviderToolChoice {
    match choice {
        ToolChoice::Auto { .. } => ProviderToolChoice::Auto,
        ToolChoice::Any { .. } => ProviderToolChoice::Required,
        ToolChoice::Tool { name, .. } => ProviderToolChoice::Specific(name.clone()),
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::anthropic::Content;
    use serde_json::json;

    #[test]
    fn test_convert_simple_text_message() {
        let request = MessagesRequest {
            model: "claude-3-haiku".to_string(),
            max_tokens: 100,
            messages: vec![Message {
                role: "user".to_string(),
                content: Content::Text("Hello".to_string()),
            }],
            system: None,
            temperature: None,
            stream: false,
            tools: None,
            tool_choice: None,
            stop_sequences: None,
        };

        let result = convert_anthropic_to_provider(&request, "openai").unwrap();

        assert_eq!(result.model, "claude-3-haiku");
        assert_eq!(result.max_tokens, 100);
        assert_eq!(result.messages.len(), 1);
        assert_eq!(result.messages[0].role, "user");
    }

    #[test]
    fn test_convert_with_system_message_text() {
        let request = MessagesRequest {
            model: "gpt-4".to_string(),
            max_tokens: 200,
            messages: vec![Message {
                role: "user".to_string(),
                content: Content::Text("Hello".to_string()),
            }],
            system: Some(SystemPrompt::Text("You are a helpful assistant".to_string())),
            temperature: None,
            stream: false,
            tools: None,
            tool_choice: None,
            stop_sequences: None,
        };

        let result = convert_anthropic_to_provider(&request, "openai").unwrap();

        assert_eq!(result.messages.len(), 2);
        assert_eq!(result.messages[0].role, "system");
        assert_eq!(result.messages[1].role, "user");
    }

    #[test]
    fn test_max_tokens_capping_openai() {
        let request = MessagesRequest {
            model: "gpt-4".to_string(),
            max_tokens: 20000, // Exceeds limit
            messages: vec![Message {
                role: "user".to_string(),
                content: Content::Text("Hello".to_string()),
            }],
            system: None,
            temperature: None,
            stream: false,
            tools: None,
            tool_choice: None,
            stop_sequences: None,
        };

        let result = convert_anthropic_to_provider(&request, "openai").unwrap();
        assert_eq!(result.max_tokens, MAX_TOKENS_LIMIT);
    }

    #[test]
    fn test_max_tokens_no_capping_ollama() {
        let request = MessagesRequest {
            model: "llama3".to_string(),
            max_tokens: 20000,
            messages: vec![Message {
                role: "user".to_string(),
                content: Content::Text("Hello".to_string()),
            }],
            system: None,
            temperature: None,
            stream: false,
            tools: None,
            tool_choice: None,
            stop_sequences: None,
        };

        let result = convert_anthropic_to_provider(&request, "ollama").unwrap();
        assert_eq!(result.max_tokens, 20000);
    }

    #[test]
    fn test_flatten_tool_result_text() {
        let result = flatten_tool_result(
            "tool_123",
            &ToolResultContent::Text {
                content: "Result content".to_string(),
            },
        );

        assert!(result.contains("tool_123"));
        assert!(result.contains("Result content"));
        assert!(result.starts_with("Tool result for"));
    }

    #[test]
    fn test_clean_schema_removes_additional_properties() {
        let schema = json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            },
            "additionalProperties": false,
        });

        let cleaned = clean_schema_for_gemini(schema).unwrap();
        assert!(cleaned.get("additionalProperties").is_none());
        assert!(cleaned.get("properties").is_some());
    }

    #[test]
    fn test_clean_schema_removes_default() {
        let schema = json!({
            "type": "string",
            "default": "hello",
        });

        let cleaned = clean_schema_for_gemini(schema).unwrap();
        assert!(cleaned.get("default").is_none());
    }

    #[test]
    fn test_clean_schema_removes_invalid_format() {
        let schema = json!({
            "type": "string",
            "format": "email",
        });

        let cleaned = clean_schema_for_gemini(schema).unwrap();
        assert!(cleaned.get("format").is_none());
    }

    #[test]
    fn test_clean_schema_keeps_enum_format() {
        let schema = json!({
            "type": "string",
            "format": "enum",
        });

        let cleaned = clean_schema_for_gemini(schema).unwrap();
        assert_eq!(cleaned.get("format").unwrap(), "enum");
    }

    #[test]
    fn test_clean_schema_recursive() {
        let schema = json!({
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "email": {
                            "type": "string",
                            "format": "email",
                            "default": "user@example.com"
                        }
                    },
                    "additionalProperties": true
                }
            }
        });

        let cleaned = clean_schema_for_gemini(schema).unwrap();

        // Check that nested properties were cleaned
        let user = &cleaned["properties"]["user"];
        assert!(user.get("additionalProperties").is_none());

        let email = &user["properties"]["email"];
        assert!(email.get("format").is_none());
        assert!(email.get("default").is_none());
    }

    #[test]
    fn test_convert_tool_choice_auto() {
        let choice = ToolChoice::Auto {
            r#type: "auto".to_string(),
        };
        let result = convert_tool_choice(&choice);
        assert!(matches!(result, ProviderToolChoice::Auto));
    }

    #[test]
    fn test_convert_tool_choice_any() {
        let choice = ToolChoice::Any {
            r#type: "any".to_string(),
        };
        let result = convert_tool_choice(&choice);
        assert!(matches!(result, ProviderToolChoice::Required));
    }

    #[test]
    fn test_convert_tool_choice_specific() {
        let choice = ToolChoice::Tool {
            r#type: "tool".to_string(),
            name: "calculator".to_string(),
        };
        let result = convert_tool_choice(&choice);

        if let ProviderToolChoice::Specific(name) = result {
            assert_eq!(name, "calculator");
        } else {
            panic!("Expected Specific tool choice");
        }
    }
}
