//! Token counting endpoint handler
//!
//! Endpoint for counting tokens in a request without making an actual completion.

use crate::conversion::anthropic_to_provider;
use crate::error::ProxyError;
use crate::handlers::messages::AppState;
use crate::models::anthropic::MessagesRequest;
use axum::{
    extract::State,
    http::StatusCode,
    Json,
};
use serde::{Deserialize, Serialize};
use tracing::info;

/// Token count request (same as MessagesRequest but without stream)
pub type TokenCountRequest = MessagesRequest;

/// Token count response
#[derive(Debug, Serialize, Deserialize)]
pub struct TokenCountResponse {
    /// Number of input tokens
    pub input_tokens: usize,
}

/// Token counting handler
///
/// Handles POST /v1/messages/count_tokens requests.
///
/// # Request Body
///
/// Same as MessagesRequest format but streaming is ignored.
///
/// # Response
///
/// ```json
/// {
///   "input_tokens": 123
/// }
/// ```
pub async fn count_tokens_handler(
    State(state): State<AppState>,
    Json(request): Json<TokenCountRequest>,
) -> Result<(StatusCode, Json<TokenCountResponse>), ProxyError> {
    info!(
        "Token count request: model={}, messages={}",
        request.model,
        request.messages.len()
    );

    // Resolve model name and determine provider
    let (resolved_model, provider_name) = resolve_model_and_provider(&state, &request.model)?;

    info!(
        "Model mapping for token count: {} → {} (provider: {})",
        request.model, resolved_model, provider_name
    );

    // Get the appropriate provider
    let provider = state.get_provider(&provider_name)?;

    // Convert request to provider format
    let mut provider_request = anthropic_to_provider::convert_anthropic_to_provider(
        &request,
        &provider_name,
    )?;

    // Update model to resolved version
    provider_request.model = resolved_model;

    // Count tokens using provider
    let input_tokens = provider.count_tokens(&provider_request).await?;

    info!("Token count result: {} tokens", input_tokens);

    // Return token count
    Ok((
        StatusCode::OK,
        Json(TokenCountResponse { input_tokens }),
    ))
}

/// Resolve model name and determine which provider to use
///
/// This is duplicated from messages.rs for now, but could be refactored
/// into a shared utility module.
fn resolve_model_and_provider(
    state: &AppState,
    model_name: &str,
) -> Result<(String, String), ProxyError> {
    // Try to resolve with each provider in order of preference
    let preferred = &state.config.providers.preferred;

    // First try the preferred provider
    if let Ok(resolved) = state.model_mapper.resolve_model(model_name, preferred) {
        let (provider, model) = strip_provider_prefix(&resolved);
        return Ok((model, provider.unwrap_or(preferred.as_str()).to_string()));
    }

    // Try other providers
    for provider in &["openai", "google", "gemini", "ollama"] {
        if *provider != preferred {
            if let Ok(resolved) = state.model_mapper.resolve_model(model_name, provider) {
                let (prov, model) = strip_provider_prefix(&resolved);
                return Ok((model, prov.unwrap_or(provider).to_string()));
            }
        }
    }

    // If no provider can handle this model, return error
    Err(ProxyError::ModelNotFound {
        model: model_name.to_string(),
        provider: preferred.clone(),
        valid_models: "Check model configuration".to_string(),
    })
}

/// Strip provider prefix from fully qualified model name
fn strip_provider_prefix(model: &str) -> (Option<&str>, String) {
    let prefixes = ["openai/", "google/", "gemini/", "ollama/", "anthropic/"];

    for prefix in &prefixes {
        if model.starts_with(prefix) {
            let provider = prefix.trim_end_matches('/');
            let base = model.strip_prefix(prefix).unwrap().to_string();
            return (Some(provider), base);
        }
    }

    (None, model.to_string())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{
        AppConfig, LoggingConfig, ModelList, ModelsConfig, OllamaDetails, ProviderDetails,
        ProvidersConfig, ServerConfig,
    };
    use crate::handlers::messages::AppState;
    use crate::models::anthropic::{Content, Message};

    fn create_test_config() -> AppConfig {
        AppConfig {
            server: ServerConfig {
                host: "127.0.0.1".to_string(),
                port: 8082,
                log_level: "info".to_string(),
                log_format: "pretty".to_string(),
            },
            providers: ProvidersConfig {
                preferred: "openai".to_string(),
                openai: ProviderDetails {
                    api_key: "test-openai-key".to_string(),
                    timeout_secs: 30,
                    max_retries: 3,
                },
                google: ProviderDetails {
                    api_key: "test-gemini-key".to_string(),
                    timeout_secs: 30,
                    max_retries: 3,
                },
                ollama: OllamaDetails {
                    api_base: "http://localhost:11434".to_string(),
                    timeout_secs: 60,
                    max_retries: 3,
                },
            },
            models: ModelsConfig {
                big: "gpt-4o".to_string(),
                small: "gpt-4o-mini".to_string(),
                openai_list: ModelList {
                    supported: vec![
                        "gpt-4o".to_string(),
                        "gpt-4o-mini".to_string(),
                        "o3-mini".to_string(),
                    ],
                },
                gemini_list: ModelList {
                    supported: vec!["gemini-2.0-flash".to_string()],
                },
                ollama_list: ModelList {
                    supported: vec!["llama3".to_string(), "mistral".to_string()],
                },
            },
            logging: LoggingConfig {
                blocked_phrases: vec![],
            },
        }
    }

    fn create_test_request() -> TokenCountRequest {
        MessagesRequest {
            model: "haiku".to_string(),
            max_tokens: 100,
            messages: vec![Message {
                role: "user".to_string(),
                content: Content::Text("Hello, world!".to_string()),
            }],
            system: None,
            temperature: None,
            stream: false,
            tools: None,
            tool_choice: None,
            stop_sequences: None,
        }
    }

    #[tokio::test]
    async fn test_count_tokens_handler_success() {
        let config = create_test_config();
        let state = AppState::new(config).unwrap();
        let request = create_test_request();

        let result = count_tokens_handler(State(state), Json(request)).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_count_tokens_handler_returns_tokens() {
        let config = create_test_config();
        let state = AppState::new(config).unwrap();
        let request = create_test_request();

        let result = count_tokens_handler(State(state), Json(request))
            .await
            .unwrap();

        let (status, Json(body)) = result;
        // Token count should be > 0 for non-empty message
        // Exact count depends on provider estimation
        assert_eq!(status, StatusCode::OK);
        assert!(body.input_tokens > 0);
    }

    #[tokio::test]
    async fn test_count_tokens_handler_unknown_model() {
        let config = create_test_config();
        let state = AppState::new(config).unwrap();
        let mut request = create_test_request();
        request.model = "unknown-model-xyz".to_string();

        let result = count_tokens_handler(State(state), Json(request)).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_count_tokens_empty_message() {
        let config = create_test_config();
        let state = AppState::new(config).unwrap();
        let mut request = create_test_request();
        request.messages[0].content = Content::Text("".to_string());

        let result = count_tokens_handler(State(state), Json(request)).await;
        assert!(result.is_ok());

        let (status, Json(body)) = result.unwrap();
        assert_eq!(status, StatusCode::OK);
        // Empty message should have 0 or very low token count
        assert_eq!(body.input_tokens, 0);
    }

    #[tokio::test]
    async fn test_count_tokens_long_message() {
        let config = create_test_config();
        let state = AppState::new(config).unwrap();
        let mut request = create_test_request();
        request.messages[0].content = Content::Text("word ".repeat(100)); // 500 chars

        let result = count_tokens_handler(State(state), Json(request)).await;
        assert!(result.is_ok());

        let (status, Json(body)) = result.unwrap();
        assert_eq!(status, StatusCode::OK);
        // 500 chars should be roughly 125 tokens (500/4)
        assert!(body.input_tokens > 100);
    }

    #[test]
    fn test_token_count_response_serialization() {
        let response = TokenCountResponse { input_tokens: 123 };
        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("\"input_tokens\":123"));
    }

    #[test]
    fn test_token_count_response_deserialization() {
        let json = r#"{"input_tokens":456}"#;
        let response: TokenCountResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.input_tokens, 456);
    }
}
