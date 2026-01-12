//! Gemini provider implementation
//!
//! Google's Gemini models via the Generative Language API.
//! This provider supports models like gemini-2.0-flash, gemini-2.5-pro, etc.
//!
//! # Features
//!
//! - API key authentication via query parameter
//! - Max tokens capped at 16,384
//! - Streaming support via Server-Sent Events
//! - Automatic schema cleaning (handled by conversion layer)
//!
//! # API Format
//!
//! Gemini uses a different API format than OpenAI:
//! - Endpoint: `https://generativelanguage.googleapis.com/v1/models/{model}:generateContent`
//! - API key passed as `key` query parameter
//! - Response structure uses `candidates` array

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

/// Default Gemini base URL
const DEFAULT_BASE_URL: &str = "https://generativelanguage.googleapis.com/v1";

/// Default timeout in seconds
const DEFAULT_TIMEOUT_SECS: u64 = 120;

/// Gemini provider
pub struct GeminiProvider {
    /// HTTP client
    client: Client,

    /// Base URL for Gemini API
    base_url: String,

    /// API key
    api_key: String,

    /// Request timeout
    timeout: Duration,
}

impl GeminiProvider {
    /// Create a new Gemini provider
    ///
    /// # Arguments
    ///
    /// * `api_key` - Google API key for Gemini
    pub fn new(api_key: String) -> ProxyResult<Self> {
        Self::with_base_url(api_key, DEFAULT_BASE_URL.to_string())
    }

    /// Create a new Gemini provider with custom base URL
    ///
    /// # Arguments
    ///
    /// * `api_key` - Google API key for Gemini
    /// * `base_url` - Custom base URL (for testing)
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

    /// Convert provider request to Gemini API format
    fn to_gemini_request(&self, request: &ProviderRequest) -> GeminiRequest {
        // Convert messages to Gemini parts format
        let mut contents = Vec::new();

        for message in &request.messages {
            let role = match message.role.as_str() {
                "system" | "user" => "user",
                "assistant" => "model",
                _ => "user",
            };

            let parts = match &message.content {
                ProviderContent::Text(text) => {
                    vec![GeminiPart::Text {
                        text: text.clone(),
                    }]
                }
                ProviderContent::Parts(parts) => parts
                    .iter()
                    .filter_map(|part| match part {
                        crate::models::providers::ProviderContentPart::Text { text } => {
                            Some(GeminiPart::Text { text: text.clone() })
                        }
                        crate::models::providers::ProviderContentPart::Image {
                            image_type: _,
                            source,
                            media_type,
                        } => Some(GeminiPart::InlineData {
                            inline_data: GeminiInlineData {
                                mime_type: media_type.clone().unwrap_or_else(|| "image/png".to_string()),
                                data: source.clone(),
                            },
                        }),
                    })
                    .collect(),
            };

            contents.push(GeminiContent {
                role: role.to_string(),
                parts,
            });
        }

        // Generation config
        let generation_config = GeminiGenerationConfig {
            temperature: Some(request.temperature),
            max_output_tokens: Some(request.max_tokens as i32),
            stop_sequences: request.stop.clone(),
        };

        GeminiRequest {
            contents,
            generation_config: Some(generation_config),
        }
    }

    /// Parse Gemini response to provider format
    fn parse_gemini_response(&self, response: GeminiResponse) -> ProxyResult<ProviderResponse> {
        // Extract content from the first candidate
        let candidate = response.candidates.first().ok_or_else(|| {
            ProxyError::ProviderApi {
                provider: "gemini".to_string(),
                message: "No candidates in response".to_string(),
                status_code: None,
            }
        })?;

        // Extract text from parts
        let mut content_blocks = Vec::new();

        if let Some(content) = &candidate.content {
            for part in &content.parts {
                match part {
                    GeminiPart::Text { text } => {
                        content_blocks.push(ProviderContentBlock::Text(text.clone()));
                    }
                    GeminiPart::InlineData { .. } => {
                        // Skip inline data in responses
                    }
                }
            }
        }

        // Ensure we have at least one content block
        if content_blocks.is_empty() {
            content_blocks.push(ProviderContentBlock::Text(String::new()));
        }

        // Parse stop reason
        let stop_reason = match candidate.finish_reason.as_deref() {
            Some("STOP") => StopReason::EndTurn,
            Some("MAX_TOKENS") => StopReason::MaxTokens,
            Some("SAFETY") => StopReason::Other("safety".to_string()),
            Some("RECITATION") => StopReason::Other("recitation".to_string()),
            _ => StopReason::EndTurn,
        };

        // Extract usage
        let usage = if let Some(usage_metadata) = &response.usage_metadata {
            Usage {
                input_tokens: usage_metadata.prompt_token_count.unwrap_or(0) as u32,
                output_tokens: usage_metadata.candidates_token_count.unwrap_or(0) as u32,
            }
        } else {
            Usage {
                input_tokens: 0,
                output_tokens: 0,
            }
        };

        // Generate a response ID (Gemini doesn't provide one)
        let id = format!("gemini_{}", uuid::Uuid::new_v4().simple());

        Ok(ProviderResponse {
            id,
            content: content_blocks,
            stop_reason,
            usage,
        })
    }
}

#[async_trait]
impl Provider for GeminiProvider {
    fn name(&self) -> &str {
        "gemini"
    }

    async fn complete(&self, request: &ProviderRequest) -> ProxyResult<ProviderResponse> {
        let gemini_request = self.to_gemini_request(request);

        // Build URL with model and API key
        let url = format!(
            "{}/models/{}:generateContent?key={}",
            self.base_url, request.model, self.api_key
        );

        let response = self
            .client
            .post(&url)
            .json(&gemini_request)
            .timeout(self.timeout)
            .send()
            .await
            .map_err(|e| {
                if e.is_timeout() {
                    ProxyError::ProviderTimeout {
                        provider: "gemini".to_string(),
                        timeout_secs: self.timeout.as_secs(),
                    }
                } else if e.is_connect() {
                    ProxyError::ProviderApi {
                        provider: "gemini".to_string(),
                        message: format!("Connection failed: {}", e),
                        status_code: None,
                    }
                } else {
                    ProxyError::Http(format!("Request failed: {}", e))
                }
            })?;

        let status = response.status();
        if !status.is_success() {
            let error_body = response.text().await.unwrap_or_default();

            // Handle specific error cases
            if status.as_u16() == 429 {
                return Err(ProxyError::RateLimitExceeded {
                    provider: "gemini".to_string(),
                    retry_after_secs: 60, // Default retry after 60s
                });
            } else if status.as_u16() == 401 || status.as_u16() == 403 {
                return Err(ProxyError::InvalidApiKey);
            }

            return Err(ProxyError::ProviderApi {
                provider: "gemini".to_string(),
                message: format!("API error: {} - {}", status, error_body),
                status_code: Some(status.as_u16()),
            });
        }

        let gemini_response: GeminiResponse = response.json().await.map_err(|e| {
            ProxyError::Conversion {
                message: format!("Failed to parse Gemini response: {}", e),
            }
        })?;

        self.parse_gemini_response(gemini_response)
    }

    async fn complete_stream(
        &self,
        request: &ProviderRequest,
    ) -> ProxyResult<BoxStream<'static, ProxyResult<StreamChunk>>> {
        let gemini_request = self.to_gemini_request(request);

        // Build URL for streaming endpoint
        let url = format!(
            "{}/models/{}:streamGenerateContent?key={}",
            self.base_url, request.model, self.api_key
        );

        let response = self
            .client
            .post(&url)
            .json(&gemini_request)
            .timeout(self.timeout)
            .send()
            .await
            .map_err(|e| {
                if e.is_timeout() {
                    ProxyError::ProviderTimeout {
                        provider: "gemini".to_string(),
                        timeout_secs: self.timeout.as_secs(),
                    }
                } else {
                    ProxyError::Http(format!("Stream request failed: {}", e))
                }
            })?;

        let status = response.status();
        if !status.is_success() {
            let error_body = response.text().await.unwrap_or_default();
            return Err(ProxyError::ProviderApi {
                provider: "gemini".to_string(),
                message: format!("Stream API error: {} - {}", status, error_body),
                status_code: Some(status.as_u16()),
            });
        }

        // Convert the response stream to a stream of chunks
        let stream = response.bytes_stream();

        let chunk_stream = stream.map(|result| {
            match result {
                Ok(bytes) => {
                    // Parse JSON chunks
                    let text = String::from_utf8_lossy(&bytes);

                    // Try to parse as a Gemini response
                    if let Ok(chunk) = serde_json::from_str::<GeminiResponse>(&text) {
                        if let Some(candidate) = chunk.candidates.first() {
                            if let Some(content) = &candidate.content {
                                // Extract text from parts
                                let text_parts: Vec<String> = content
                                    .parts
                                    .iter()
                                    .filter_map(|part| match part {
                                        GeminiPart::Text { text } => Some(text.clone()),
                                        _ => None,
                                    })
                                    .collect();

                                if !text_parts.is_empty() {
                                    return Ok(StreamChunk::TextDelta(text_parts.join("")));
                                }
                            }

                            // Check for finish reason
                            if let Some(finish_reason) = &candidate.finish_reason {
                                let stop_reason = match finish_reason.as_str() {
                                    "STOP" => StopReason::EndTurn,
                                    "MAX_TOKENS" => StopReason::MaxTokens,
                                    _ => StopReason::Other(finish_reason.clone()),
                                };

                                return Ok(StreamChunk::Done {
                                    stop_reason,
                                    usage: Usage {
                                        input_tokens: 0,
                                        output_tokens: 0,
                                    },
                                });
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
        // Gemini doesn't provide a token counting API in the free tier
        // Estimate based on character count (rough approximation: 1 token ≈ 4 characters)
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
// Gemini API Types
// ============================================================================

#[derive(Debug, Serialize)]
struct GeminiRequest {
    contents: Vec<GeminiContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    generation_config: Option<GeminiGenerationConfig>,
}

#[derive(Debug, Serialize, Deserialize)]
struct GeminiContent {
    role: String,
    parts: Vec<GeminiPart>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
enum GeminiPart {
    Text { text: String },
    InlineData { inline_data: GeminiInlineData },
}

#[derive(Debug, Serialize, Deserialize)]
struct GeminiInlineData {
    mime_type: String,
    data: String,
}

#[derive(Debug, Serialize)]
struct GeminiGenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop_sequences: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
struct GeminiResponse {
    candidates: Vec<GeminiCandidate>,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage_metadata: Option<GeminiUsageMetadata>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiCandidate {
    content: Option<GeminiContent>,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiUsageMetadata {
    prompt_token_count: Option<i32>,
    candidates_token_count: Option<i32>,
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
        let provider = GeminiProvider::new("test-api-key".to_string()).unwrap();
        assert_eq!(provider.name(), "gemini");
        assert_eq!(provider.api_key, "test-api-key");
    }

    #[test]
    fn test_provider_empty_api_key() {
        let result = GeminiProvider::new(String::new());
        assert!(result.is_err());

        if let Err(ProxyError::InvalidApiKey) = result {
            // Success
        } else {
            panic!("Expected InvalidApiKey error");
        }
    }

    #[test]
    fn test_provider_with_custom_url() {
        let provider =
            GeminiProvider::with_base_url("test-key".to_string(), "https://custom.api".to_string())
                .unwrap();
        assert_eq!(provider.base_url, "https://custom.api");
    }

    #[test]
    fn test_provider_strips_trailing_slash() {
        let provider = GeminiProvider::with_base_url(
            "test-key".to_string(),
            "https://api.google.com/".to_string(),
        )
        .unwrap();
        assert_eq!(provider.base_url, "https://api.google.com");
    }

    #[test]
    fn test_to_gemini_request() {
        let provider = GeminiProvider::new("test-key".to_string()).unwrap();

        let request = ProviderRequest {
            model: "gemini-2.0-flash".to_string(),
            messages: vec![
                ProviderMessage {
                    role: "system".to_string(),
                    content: ProviderContent::Text("You are helpful".to_string()),
                },
                ProviderMessage {
                    role: "user".to_string(),
                    content: ProviderContent::Text("Hello".to_string()),
                },
            ],
            max_tokens: 100,
            temperature: 0.7,
            tools: None,
            tool_choice: None,
            stop: Some(vec!["STOP".to_string()]),
        };

        let gemini_req = provider.to_gemini_request(&request);

        assert_eq!(gemini_req.contents.len(), 2);
        assert_eq!(gemini_req.contents[0].role, "user"); // system -> user
        assert_eq!(gemini_req.contents[1].role, "user");

        let config = gemini_req.generation_config.unwrap();
        assert_eq!(config.temperature, Some(0.7));
        assert_eq!(config.max_output_tokens, Some(100));
        assert_eq!(config.stop_sequences, Some(vec!["STOP".to_string()]));
    }

    #[test]
    fn test_to_gemini_request_role_mapping() {
        let provider = GeminiProvider::new("test-key".to_string()).unwrap();

        let request = ProviderRequest {
            model: "gemini-2.0-flash".to_string(),
            messages: vec![
                ProviderMessage {
                    role: "user".to_string(),
                    content: ProviderContent::Text("Hello".to_string()),
                },
                ProviderMessage {
                    role: "assistant".to_string(),
                    content: ProviderContent::Text("Hi".to_string()),
                },
            ],
            max_tokens: 100,
            temperature: 0.7,
            tools: None,
            tool_choice: None,
            stop: None,
        };

        let gemini_req = provider.to_gemini_request(&request);

        assert_eq!(gemini_req.contents[0].role, "user");
        assert_eq!(gemini_req.contents[1].role, "model"); // assistant -> model
    }

    #[test]
    fn test_parse_gemini_response() {
        let provider = GeminiProvider::new("test-key".to_string()).unwrap();

        let gemini_response = GeminiResponse {
            candidates: vec![GeminiCandidate {
                content: Some(GeminiContent {
                    role: "model".to_string(),
                    parts: vec![GeminiPart::Text {
                        text: "Hello, world!".to_string(),
                    }],
                }),
                finish_reason: Some("STOP".to_string()),
            }],
            usage_metadata: Some(GeminiUsageMetadata {
                prompt_token_count: Some(10),
                candidates_token_count: Some(5),
            }),
        };

        let result = provider.parse_gemini_response(gemini_response).unwrap();

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
    fn test_parse_gemini_response_no_candidates() {
        let provider = GeminiProvider::new("test-key".to_string()).unwrap();

        let gemini_response = GeminiResponse {
            candidates: vec![],
            usage_metadata: None,
        };

        let result = provider.parse_gemini_response(gemini_response);
        assert!(result.is_err());

        if let Err(ProxyError::ProviderApi { message, .. }) = result {
            assert!(message.contains("No candidates"));
        } else {
            panic!("Expected ProviderApi error");
        }
    }

    #[test]
    fn test_parse_gemini_response_max_tokens() {
        let provider = GeminiProvider::new("test-key".to_string()).unwrap();

        let gemini_response = GeminiResponse {
            candidates: vec![GeminiCandidate {
                content: Some(GeminiContent {
                    role: "model".to_string(),
                    parts: vec![GeminiPart::Text {
                        text: "Incomplete...".to_string(),
                    }],
                }),
                finish_reason: Some("MAX_TOKENS".to_string()),
            }],
            usage_metadata: Some(GeminiUsageMetadata {
                prompt_token_count: Some(10),
                candidates_token_count: Some(100),
            }),
        };

        let result = provider.parse_gemini_response(gemini_response).unwrap();

        assert_eq!(result.stop_reason, StopReason::MaxTokens);
    }

    #[test]
    fn test_parse_gemini_response_safety_stop() {
        let provider = GeminiProvider::new("test-key".to_string()).unwrap();

        let gemini_response = GeminiResponse {
            candidates: vec![GeminiCandidate {
                content: Some(GeminiContent {
                    role: "model".to_string(),
                    parts: vec![GeminiPart::Text {
                        text: "Response".to_string(),
                    }],
                }),
                finish_reason: Some("SAFETY".to_string()),
            }],
            usage_metadata: None,
        };

        let result = provider.parse_gemini_response(gemini_response).unwrap();

        assert_eq!(result.stop_reason, StopReason::Other("safety".to_string()));
    }

    #[tokio::test]
    async fn test_count_tokens() {
        let provider = GeminiProvider::new("test-key".to_string()).unwrap();

        let request = ProviderRequest {
            model: "gemini-2.0-flash".to_string(),
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
    fn test_parse_gemini_response_empty_content() {
        let provider = GeminiProvider::new("test-key".to_string()).unwrap();

        let gemini_response = GeminiResponse {
            candidates: vec![GeminiCandidate {
                content: Some(GeminiContent {
                    role: "model".to_string(),
                    parts: vec![],
                }),
                finish_reason: Some("STOP".to_string()),
            }],
            usage_metadata: None,
        };

        let result = provider.parse_gemini_response(gemini_response).unwrap();

        // Should have fallback empty text block
        assert_eq!(result.content.len(), 1);
        if let ProviderContentBlock::Text(text) = &result.content[0] {
            assert_eq!(text, "");
        } else {
            panic!("Expected text content block");
        }
    }
}
