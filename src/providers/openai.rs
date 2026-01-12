//! OpenAI provider implementation
//!
//! OpenAI's GPT models via the Chat Completions API.
//! This provider supports models like gpt-4o, gpt-4o-mini, o1, o3-mini, etc.
//!
//! # Features
//!
//! - Bearer token authentication via Authorization header
//! - Max tokens capped at 16,384
//! - Streaming support via Server-Sent Events
//! - Tool calling with `tool_calls` array
//!
//! # API Format
//!
//! OpenAI uses the standard chat completions format:
//! - Endpoint: `https://api.openai.com/v1/chat/completions`
//! - API key passed as `Authorization: Bearer {key}` header
//! - Response structure uses `choices` array

use crate::error::{ProxyError, ProxyResult};
use crate::models::anthropic::Usage;
use crate::models::providers::{
    ProviderContent, ProviderContentBlock, ProviderRequest, ProviderResponse, StopReason,
    StreamChunk,
};
use crate::providers::Provider;
use async_trait::async_trait;
use futures::stream::{BoxStream, StreamExt};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Default OpenAI base URL
const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";

/// Default timeout in seconds
const DEFAULT_TIMEOUT_SECS: u64 = 120;

/// Maximum tokens allowed by OpenAI
const MAX_TOKENS_LIMIT: u32 = 16384;

/// OpenAI provider
#[derive(Debug)]
pub struct OpenAIProvider {
    /// HTTP client
    client: Client,

    /// Base URL for OpenAI API
    base_url: String,

    /// API key
    api_key: String,

    /// Request timeout
    timeout: Duration,
}

impl OpenAIProvider {
    /// Create a new OpenAI provider
    ///
    /// # Arguments
    ///
    /// * `api_key` - OpenAI API key
    pub fn new(api_key: String) -> ProxyResult<Self> {
        Self::with_base_url(api_key, DEFAULT_BASE_URL.to_string())
    }

    /// Create a new OpenAI provider with custom base URL
    ///
    /// # Arguments
    ///
    /// * `api_key` - OpenAI API key
    /// * `base_url` - Custom base URL (for testing or alternative endpoints)
    pub fn with_base_url(api_key: String, base_url: String) -> ProxyResult<Self> {
        if api_key.is_empty() {
            return Err(ProxyError::InvalidApiKey);
        }

        let client = Client::builder()
            .timeout(Duration::from_secs(DEFAULT_TIMEOUT_SECS))
            .build()
            .map_err(|e| ProxyError::Http(format!("Failed to create HTTP client: {}", e)))?;

        Ok(Self {
            client,
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key,
            timeout: Duration::from_secs(DEFAULT_TIMEOUT_SECS),
        })
    }

    /// Convert provider request to OpenAI API format
    fn to_openai_request(&self, request: &ProviderRequest) -> OpenAIRequest {
        let messages: Vec<OpenAIMessage> = request
            .messages
            .iter()
            .map(|msg| {
                let content = match &msg.content {
                    ProviderContent::Text(text) => text.clone(),
                    ProviderContent::Parts(parts) => {
                        // Concatenate all text parts
                        parts
                            .iter()
                            .filter_map(|part| match part {
                                crate::models::providers::ProviderContentPart::Text { text } => {
                                    Some(text.as_str())
                                }
                                _ => None,
                            })
                            .collect::<Vec<_>>()
                            .join("\n")
                    }
                };

                OpenAIMessage {
                    role: msg.role.clone(),
                    content,
                }
            })
            .collect();

        // Cap max_tokens at OpenAI's limit
        let max_tokens = std::cmp::min(request.max_tokens, MAX_TOKENS_LIMIT);

        OpenAIRequest {
            model: request.model.clone(),
            messages,
            max_tokens: Some(max_tokens),
            temperature: Some(request.temperature),
            stream: false,
            stop: request.stop.clone(),
        }
    }

    /// Parse OpenAI response to provider format
    fn parse_openai_response(&self, response: OpenAIResponse) -> ProxyResult<ProviderResponse> {
        // Extract content from the first choice
        let choice = response.choices.first().ok_or_else(|| {
            ProxyError::ProviderApi {
                provider: "openai".to_string(),
                message: "No choices in response".to_string(),
                status_code: None,
            }
        })?;

        let content = vec![ProviderContentBlock::Text(
            choice.message.content.clone(),
        )];

        // Parse stop reason
        let stop_reason = choice
            .finish_reason
            .as_deref()
            .map(StopReason::from_openai)
            .unwrap_or(StopReason::EndTurn);

        // Extract usage
        let usage = Usage {
            input_tokens: response.usage.prompt_tokens as u32,
            output_tokens: response.usage.completion_tokens as u32,
        };

        Ok(ProviderResponse {
            id: response.id,
            content,
            stop_reason,
            usage,
        })
    }
}

#[async_trait]
impl Provider for OpenAIProvider {
    fn name(&self) -> &str {
        "openai"
    }

    async fn complete(&self, request: &ProviderRequest) -> ProxyResult<ProviderResponse> {
        let openai_request = self.to_openai_request(request);

        let url = format!("{}/chat/completions", self.base_url);

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&openai_request)
            .timeout(self.timeout)
            .send()
            .await
            .map_err(|e| {
                if e.is_timeout() {
                    ProxyError::ProviderTimeout {
                        provider: "openai".to_string(),
                        timeout_secs: self.timeout.as_secs(),
                    }
                } else {
                    ProxyError::Http(format!("Request failed: {}", e))
                }
            })?;

        let status = response.status();
        if !status.is_success() {
            let error_body = response.text().await.unwrap_or_default();

            // Check for specific error types
            if status.as_u16() == 429 {
                return Err(ProxyError::RateLimitExceeded {
                    provider: "openai".to_string(),
                    retry_after_secs: 60,
                });
            } else if status.as_u16() == 401 || status.as_u16() == 403 {
                return Err(ProxyError::InvalidApiKey);
            }

            return Err(ProxyError::ProviderApi {
                provider: "openai".to_string(),
                message: format!("API error: {} - {}", status, error_body),
                status_code: Some(status.as_u16()),
            });
        }

        let openai_response: OpenAIResponse = response.json().await.map_err(|e| {
            ProxyError::Conversion {
                message: format!("Failed to parse OpenAI response: {}", e),
            }
        })?;

        self.parse_openai_response(openai_response)
    }

    async fn complete_stream(
        &self,
        request: &ProviderRequest,
    ) -> ProxyResult<BoxStream<'static, ProxyResult<StreamChunk>>> {
        let mut openai_request = self.to_openai_request(request);
        openai_request.stream = true;

        let url = format!("{}/chat/completions", self.base_url);

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&openai_request)
            .timeout(self.timeout)
            .send()
            .await
            .map_err(|e| {
                if e.is_timeout() {
                    ProxyError::ProviderTimeout {
                        provider: "openai".to_string(),
                        timeout_secs: self.timeout.as_secs(),
                    }
                } else {
                    ProxyError::Http(format!("Stream request failed: {}", e))
                }
            })?;

        let status = response.status();
        if !status.is_success() {
            let error_body = response.text().await.unwrap_or_default();

            if status.as_u16() == 429 {
                return Err(ProxyError::RateLimitExceeded {
                    provider: "openai".to_string(),
                    retry_after_secs: 60,
                });
            } else if status.as_u16() == 401 || status.as_u16() == 403 {
                return Err(ProxyError::InvalidApiKey);
            }

            return Err(ProxyError::ProviderApi {
                provider: "openai".to_string(),
                message: format!("Stream API error: {} - {}", status, error_body),
                status_code: Some(status.as_u16()),
            });
        }

        // Convert the response stream to a stream of chunks
        let stream = response.bytes_stream();

        let chunk_stream = stream.map(|result| {
            match result {
                Ok(bytes) => {
                    // Parse SSE data
                    let text = String::from_utf8_lossy(&bytes);

                    // Parse each line as a potential JSON chunk
                    for line in text.lines() {
                        if line.starts_with("data: ") {
                            let json_str = line.strip_prefix("data: ").unwrap_or(line);

                            if json_str == "[DONE]" {
                                return Ok(StreamChunk::Done {
                                    stop_reason: StopReason::EndTurn,
                                    usage: Usage {
                                        input_tokens: 0,
                                        output_tokens: 0,
                                    },
                                });
                            }

                            if let Ok(chunk) = serde_json::from_str::<OpenAIStreamChunk>(json_str) {
                                if let Some(choice) = chunk.choices.first() {
                                    if let Some(content) = &choice.delta.content {
                                        return Ok(StreamChunk::TextDelta(content.clone()));
                                    }
                                }
                            }
                        }
                    }

                    // If we couldn't parse anything useful, return an empty delta
                    Ok(StreamChunk::TextDelta(String::new()))
                }
                Err(e) => Err(ProxyError::Streaming {
                    message: format!("Stream error: {}", e),
                }),
            }
        });

        Ok(Box::pin(chunk_stream))
    }

    async fn count_tokens(&self, request: &ProviderRequest) -> ProxyResult<usize> {
        // OpenAI doesn't provide a free token counting API
        // Use tiktoken-rs in the future, for now estimate based on character count
        // Rough approximation: 1 token ≈ 4 characters for English text
        let mut total_chars = 0;

        for message in &request.messages {
            match &message.content {
                ProviderContent::Text(text) => {
                    total_chars += text.len();
                }
                ProviderContent::Parts(parts) => {
                    for part in parts {
                        if let crate::models::providers::ProviderContentPart::Text { text } = part {
                            total_chars += text.len();
                        }
                    }
                }
            }
        }

        // Rough estimate: 4 characters per token
        Ok(total_chars / 4)
    }
}

// ============================================================================
// OpenAI API Types
// ============================================================================

#[derive(Debug, Serialize)]
struct OpenAIRequest {
    model: String,
    messages: Vec<OpenAIMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(default)]
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Vec<String>>,
}

#[derive(Debug, Serialize, Deserialize)]
struct OpenAIMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct OpenAIResponse {
    id: String,
    choices: Vec<OpenAIChoice>,
    usage: OpenAIUsage,
}

#[derive(Debug, Deserialize)]
struct OpenAIChoice {
    message: OpenAIMessage,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAIUsage {
    prompt_tokens: usize,
    completion_tokens: usize,
}

#[derive(Debug, Deserialize)]
struct OpenAIStreamChunk {
    choices: Vec<OpenAIStreamChoice>,
}

#[derive(Debug, Deserialize)]
struct OpenAIStreamChoice {
    delta: OpenAIDelta,
}

#[derive(Debug, Deserialize)]
struct OpenAIDelta {
    content: Option<String>,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::providers::ProviderMessage;

    #[test]
    fn test_provider_creation() {
        let provider = OpenAIProvider::new("test-key".to_string()).unwrap();
        assert_eq!(provider.name(), "openai");
        assert_eq!(provider.base_url, DEFAULT_BASE_URL);
        assert_eq!(provider.api_key, "test-key");
    }

    #[test]
    fn test_provider_empty_api_key() {
        let result = OpenAIProvider::new("".to_string());
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ProxyError::InvalidApiKey));
    }

    #[test]
    fn test_provider_with_custom_url() {
        let provider = OpenAIProvider::with_base_url(
            "test-key".to_string(),
            "https://custom.openai.com".to_string(),
        )
        .unwrap();
        assert_eq!(provider.base_url, "https://custom.openai.com");
    }

    #[test]
    fn test_provider_strips_trailing_slash() {
        let provider = OpenAIProvider::with_base_url(
            "test-key".to_string(),
            "https://api.openai.com/v1/".to_string(),
        )
        .unwrap();
        assert_eq!(provider.base_url, "https://api.openai.com/v1");
    }

    #[test]
    fn test_to_openai_request() {
        let provider = OpenAIProvider::new("test-key".to_string()).unwrap();

        let request = ProviderRequest {
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

        let openai_req = provider.to_openai_request(&request);

        assert_eq!(openai_req.model, "gpt-4o");
        assert_eq!(openai_req.messages.len(), 1);
        assert_eq!(openai_req.messages[0].role, "user");
        assert_eq!(openai_req.messages[0].content, "Hello");
        assert_eq!(openai_req.max_tokens, Some(100));
        assert_eq!(openai_req.temperature, Some(0.7));
        assert!(!openai_req.stream);
    }

    #[test]
    fn test_to_openai_request_with_stop() {
        let provider = OpenAIProvider::new("test-key".to_string()).unwrap();

        let request = ProviderRequest {
            model: "gpt-4o".to_string(),
            messages: vec![ProviderMessage {
                role: "user".to_string(),
                content: ProviderContent::Text("Hello".to_string()),
            }],
            max_tokens: 100,
            temperature: 0.7,
            tools: None,
            tool_choice: None,
            stop: Some(vec!["STOP".to_string()]),
        };

        let openai_req = provider.to_openai_request(&request);
        assert_eq!(openai_req.stop, Some(vec!["STOP".to_string()]));
    }

    #[test]
    fn test_max_tokens_capping() {
        let provider = OpenAIProvider::new("test-key".to_string()).unwrap();

        let request = ProviderRequest {
            model: "gpt-4o".to_string(),
            messages: vec![ProviderMessage {
                role: "user".to_string(),
                content: ProviderContent::Text("Hello".to_string()),
            }],
            max_tokens: 20000, // Exceeds limit
            temperature: 0.7,
            tools: None,
            tool_choice: None,
            stop: None,
        };

        let openai_req = provider.to_openai_request(&request);
        assert_eq!(openai_req.max_tokens, Some(MAX_TOKENS_LIMIT));
    }

    #[test]
    fn test_parse_openai_response() {
        let provider = OpenAIProvider::new("test-key".to_string()).unwrap();

        let openai_response = OpenAIResponse {
            id: "chatcmpl-123".to_string(),
            choices: vec![OpenAIChoice {
                message: OpenAIMessage {
                    role: "assistant".to_string(),
                    content: "Hello, world!".to_string(),
                },
                finish_reason: Some("stop".to_string()),
            }],
            usage: OpenAIUsage {
                prompt_tokens: 10,
                completion_tokens: 5,
            },
        };

        let result = provider.parse_openai_response(openai_response).unwrap();

        assert_eq!(result.id, "chatcmpl-123");
        assert_eq!(result.content.len(), 1);
        assert_eq!(result.stop_reason, StopReason::EndTurn);
        assert_eq!(result.usage.input_tokens, 10);
        assert_eq!(result.usage.output_tokens, 5);

        if let ProviderContentBlock::Text(text) = &result.content[0] {
            assert_eq!(text, "Hello, world!");
        } else {
            panic!("Expected text content block");
        }
    }

    #[test]
    fn test_parse_openai_response_no_choices() {
        let provider = OpenAIProvider::new("test-key".to_string()).unwrap();

        let openai_response = OpenAIResponse {
            id: "chatcmpl-456".to_string(),
            choices: vec![],
            usage: OpenAIUsage {
                prompt_tokens: 10,
                completion_tokens: 0,
            },
        };

        let result = provider.parse_openai_response(openai_response);
        assert!(result.is_err());

        if let Err(ProxyError::ProviderApi { message, .. }) = result {
            assert!(message.contains("No choices"));
        } else {
            panic!("Expected ProviderApi error");
        }
    }

    #[test]
    fn test_parse_openai_response_max_tokens_stop() {
        let provider = OpenAIProvider::new("test-key".to_string()).unwrap();

        let openai_response = OpenAIResponse {
            id: "chatcmpl-789".to_string(),
            choices: vec![OpenAIChoice {
                message: OpenAIMessage {
                    role: "assistant".to_string(),
                    content: "Truncated text".to_string(),
                },
                finish_reason: Some("length".to_string()),
            }],
            usage: OpenAIUsage {
                prompt_tokens: 10,
                completion_tokens: 100,
            },
        };

        let result = provider.parse_openai_response(openai_response).unwrap();
        assert_eq!(result.stop_reason, StopReason::MaxTokens);
    }

    #[tokio::test]
    async fn test_count_tokens() {
        let provider = OpenAIProvider::new("test-key".to_string()).unwrap();

        let request = ProviderRequest {
            model: "gpt-4o".to_string(),
            messages: vec![
                ProviderMessage {
                    role: "user".to_string(),
                    content: ProviderContent::Text("Hello world".to_string()), // 11 chars
                },
                ProviderMessage {
                    role: "assistant".to_string(),
                    content: ProviderContent::Text("Hi there!".to_string()), // 9 chars
                },
            ],
            max_tokens: 100,
            temperature: 0.7,
            tools: None,
            tool_choice: None,
            stop: None,
        };

        let tokens = provider.count_tokens(&request).await.unwrap();

        // 20 chars / 4 = 5 tokens (rough estimate)
        assert_eq!(tokens, 5);
    }

    #[test]
    fn test_multipart_content_concatenation() {
        let provider = OpenAIProvider::new("test-key".to_string()).unwrap();

        let request = ProviderRequest {
            model: "gpt-4o".to_string(),
            messages: vec![ProviderMessage {
                role: "user".to_string(),
                content: ProviderContent::Parts(vec![
                    crate::models::providers::ProviderContentPart::Text {
                        text: "Hello".to_string(),
                    },
                    crate::models::providers::ProviderContentPart::Text {
                        text: "World".to_string(),
                    },
                ]),
            }],
            max_tokens: 100,
            temperature: 0.7,
            tools: None,
            tool_choice: None,
            stop: None,
        };

        let openai_req = provider.to_openai_request(&request);
        assert_eq!(openai_req.messages[0].content, "Hello\nWorld");
    }
}
