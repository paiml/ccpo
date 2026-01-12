//! Ollama provider implementation
//!
//! Ollama provides local LLM inference with an OpenAI-compatible API.
//! This provider supports models like llama3, mistral, mixtral, etc.
//!
//! # Features
//!
//! - OpenAI-compatible API format
//! - No API key required (local server)
//! - Configurable base URL (default: http://localhost:11434)
//! - No max tokens limit
//! - Streaming support via Server-Sent Events

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

/// Default Ollama base URL
const DEFAULT_BASE_URL: &str = "http://localhost:11434";

/// Default timeout in seconds
const DEFAULT_TIMEOUT_SECS: u64 = 120;

/// Ollama provider
pub struct OllamaProvider {
    /// HTTP client
    client: Client,

    /// Base URL for Ollama API
    base_url: String,

    /// Request timeout
    timeout: Duration,
}

impl OllamaProvider {
    /// Create a new Ollama provider with default settings
    pub fn new() -> ProxyResult<Self> {
        Self::with_base_url(DEFAULT_BASE_URL.to_string())
    }

    /// Create a new Ollama provider with custom base URL
    pub fn with_base_url(base_url: String) -> ProxyResult<Self> {
        let client = Client::builder()
            .timeout(Duration::from_secs(DEFAULT_TIMEOUT_SECS))
            .build()
            .map_err(|e| ProxyError::Http(format!("Failed to create HTTP client: {}", e)))?;

        Ok(Self {
            client,
            base_url: base_url.trim_end_matches('/').to_string(),
            timeout: Duration::from_secs(DEFAULT_TIMEOUT_SECS),
        })
    }

    /// Convert provider request to Ollama API format (OpenAI-compatible)
    fn to_ollama_request(&self, request: &ProviderRequest) -> OllamaRequest {
        let messages: Vec<OllamaMessage> = request
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

                OllamaMessage {
                    role: msg.role.clone(),
                    content,
                }
            })
            .collect();

        OllamaRequest {
            model: request.model.clone(),
            messages,
            max_tokens: Some(request.max_tokens),
            temperature: Some(request.temperature),
            stream: false,
            stop: request.stop.clone(),
        }
    }

    /// Parse Ollama response to provider format
    fn parse_ollama_response(&self, response: OllamaResponse) -> ProxyResult<ProviderResponse> {
        // Extract content from the first choice
        let choice = response.choices.first().ok_or_else(|| {
            ProxyError::ProviderApi {
                provider: "ollama".to_string(),
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

impl Default for OllamaProvider {
    fn default() -> Self {
        Self::new().expect("Failed to create default Ollama provider")
    }
}

#[async_trait]
impl Provider for OllamaProvider {
    fn name(&self) -> &str {
        "ollama"
    }

    async fn complete(&self, request: &ProviderRequest) -> ProxyResult<ProviderResponse> {
        let ollama_request = self.to_ollama_request(request);

        let url = format!("{}/v1/chat/completions", self.base_url);

        let response = self
            .client
            .post(&url)
            .json(&ollama_request)
            .timeout(self.timeout)
            .send()
            .await
            .map_err(|e| {
                if e.is_timeout() {
                    ProxyError::ProviderTimeout {
                        provider: "ollama".to_string(),
                        timeout_secs: self.timeout.as_secs(),
                    }
                } else if e.is_connect() {
                    ProxyError::ProviderApi {
                        provider: "ollama".to_string(),
                        message: format!("Connection failed: {}. Is Ollama running?", e),
                        status_code: None,
                    }
                } else {
                    ProxyError::Http(format!("Request failed: {}", e))
                }
            })?;

        let status = response.status();
        if !status.is_success() {
            let error_body = response.text().await.unwrap_or_default();
            return Err(ProxyError::ProviderApi {
                provider: "ollama".to_string(),
                message: format!("API error: {} - {}", status, error_body),
                status_code: Some(status.as_u16()),
            });
        }

        let ollama_response: OllamaResponse = response.json().await.map_err(|e| {
            ProxyError::Conversion {
                message: format!("Failed to parse Ollama response: {}", e),
            }
        })?;

        self.parse_ollama_response(ollama_response)
    }

    async fn complete_stream(
        &self,
        request: &ProviderRequest,
    ) -> ProxyResult<BoxStream<'static, ProxyResult<StreamChunk>>> {
        let mut ollama_request = self.to_ollama_request(request);
        ollama_request.stream = true;

        let url = format!("{}/v1/chat/completions", self.base_url);

        let response = self
            .client
            .post(&url)
            .json(&ollama_request)
            .timeout(self.timeout)
            .send()
            .await
            .map_err(|e| {
                if e.is_timeout() {
                    ProxyError::ProviderTimeout {
                        provider: "ollama".to_string(),
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
                provider: "ollama".to_string(),
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

                            if let Ok(chunk) = serde_json::from_str::<OllamaStreamChunk>(json_str) {
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
        // Ollama doesn't provide a token counting API
        // Estimate based on word count (rough approximation: 1 token ≈ 0.75 words)
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
// Ollama API Types (OpenAI-compatible)
// ============================================================================

#[derive(Debug, Serialize)]
struct OllamaRequest {
    model: String,
    messages: Vec<OllamaMessage>,
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
struct OllamaMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct OllamaResponse {
    id: String,
    choices: Vec<OllamaChoice>,
    usage: OllamaUsage,
}

#[derive(Debug, Deserialize)]
struct OllamaChoice {
    message: OllamaMessage,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OllamaUsage {
    prompt_tokens: usize,
    completion_tokens: usize,
}

#[derive(Debug, Deserialize)]
struct OllamaStreamChunk {
    choices: Vec<OllamaStreamChoice>,
}

#[derive(Debug, Deserialize)]
struct OllamaStreamChoice {
    delta: OllamaDelta,
}

#[derive(Debug, Deserialize)]
struct OllamaDelta {
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
        let provider = OllamaProvider::new().unwrap();
        assert_eq!(provider.name(), "ollama");
        assert_eq!(provider.base_url, DEFAULT_BASE_URL);
    }

    #[test]
    fn test_provider_with_custom_url() {
        let provider = OllamaProvider::with_base_url("http://custom:8080".to_string()).unwrap();
        assert_eq!(provider.base_url, "http://custom:8080");
    }

    #[test]
    fn test_provider_strips_trailing_slash() {
        let provider =
            OllamaProvider::with_base_url("http://localhost:11434/".to_string()).unwrap();
        assert_eq!(provider.base_url, "http://localhost:11434");
    }

    #[test]
    fn test_to_ollama_request() {
        let provider = OllamaProvider::new().unwrap();

        let request = ProviderRequest {
            model: "llama3".to_string(),
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

        let ollama_req = provider.to_ollama_request(&request);

        assert_eq!(ollama_req.model, "llama3");
        assert_eq!(ollama_req.messages.len(), 1);
        assert_eq!(ollama_req.messages[0].role, "user");
        assert_eq!(ollama_req.messages[0].content, "Hello");
        assert_eq!(ollama_req.max_tokens, Some(100));
        assert_eq!(ollama_req.temperature, Some(0.7));
        assert!(!ollama_req.stream);
    }

    #[test]
    fn test_to_ollama_request_with_stop() {
        let provider = OllamaProvider::new().unwrap();

        let request = ProviderRequest {
            model: "llama3".to_string(),
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

        let ollama_req = provider.to_ollama_request(&request);
        assert_eq!(ollama_req.stop, Some(vec!["STOP".to_string()]));
    }

    #[test]
    fn test_parse_ollama_response() {
        let provider = OllamaProvider::new().unwrap();

        let ollama_response = OllamaResponse {
            id: "resp_123".to_string(),
            choices: vec![OllamaChoice {
                message: OllamaMessage {
                    role: "assistant".to_string(),
                    content: "Hello, world!".to_string(),
                },
                finish_reason: Some("stop".to_string()),
            }],
            usage: OllamaUsage {
                prompt_tokens: 10,
                completion_tokens: 5,
            },
        };

        let result = provider.parse_ollama_response(ollama_response).unwrap();

        assert_eq!(result.id, "resp_123");
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
    fn test_parse_ollama_response_no_choices() {
        let provider = OllamaProvider::new().unwrap();

        let ollama_response = OllamaResponse {
            id: "resp_456".to_string(),
            choices: vec![],
            usage: OllamaUsage {
                prompt_tokens: 10,
                completion_tokens: 0,
            },
        };

        let result = provider.parse_ollama_response(ollama_response);
        assert!(result.is_err());

        if let Err(ProxyError::ProviderApi { message, .. }) = result {
            assert!(message.contains("No choices"));
        } else {
            panic!("Expected ProviderApi error");
        }
    }

    #[tokio::test]
    async fn test_count_tokens() {
        let provider = OllamaProvider::new().unwrap();

        let request = ProviderRequest {
            model: "llama3".to_string(),
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
    fn test_default_provider() {
        let provider = OllamaProvider::default();
        assert_eq!(provider.name(), "ollama");
    }
}
