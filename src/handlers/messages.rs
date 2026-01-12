//! Messages endpoint handler
//!
//! Main completion endpoint that handles POST /v1/messages requests.
//! Supports both streaming and non-streaming responses.

use crate::config::AppConfig;
use crate::conversion::{anthropic_to_provider, provider_to_anthropic};
use crate::error::{ProxyError, ProxyResult};
use crate::mapping::ModelMapper;
use crate::models::anthropic::MessagesRequest;
use crate::providers::{gemini::GeminiProvider, ollama::OllamaProvider, openai::OpenAIProvider, Provider};
use axum::{
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use std::sync::Arc;
use tracing::{info, warn};

/// Application state shared across handlers
#[derive(Clone)]
pub struct AppState {
    /// Application configuration
    pub config: Arc<AppConfig>,

    /// OpenAI provider instance
    pub openai_provider: Option<Arc<OpenAIProvider>>,

    /// Gemini provider instance
    pub gemini_provider: Option<Arc<GeminiProvider>>,

    /// Ollama provider instance
    pub ollama_provider: Option<Arc<OllamaProvider>>,

    /// Model mapper
    pub model_mapper: Arc<ModelMapper>,
}

impl AppState {
    /// Create new application state
    pub fn new(config: AppConfig) -> ProxyResult<Self> {
        // Create provider instances based on configuration
        let openai_provider = if !config.providers.openai.api_key.is_empty() {
            Some(Arc::new(OpenAIProvider::new(
                config.providers.openai.api_key.clone(),
            )?))
        } else {
            None
        };

        let gemini_provider = if !config.providers.google.api_key.is_empty() {
            Some(Arc::new(GeminiProvider::new(
                config.providers.google.api_key.clone(),
            )?))
        } else {
            None
        };

        let ollama_provider = Some(Arc::new(OllamaProvider::new()?));

        let model_mapper = Arc::new(ModelMapper::new(config.models.clone()));

        Ok(Self {
            config: Arc::new(config),
            openai_provider,
            gemini_provider,
            ollama_provider,
            model_mapper,
        })
    }

    /// Get provider by name
    pub fn get_provider(&self, provider_name: &str) -> ProxyResult<Arc<dyn Provider>> {
        match provider_name {
            "openai" => self
                .openai_provider
                .as_ref()
                .map(|p| p.clone() as Arc<dyn Provider>)
                .ok_or_else(|| ProxyError::ProviderApi {
                    provider: "openai".to_string(),
                    message: "OpenAI provider not configured".to_string(),
                    status_code: None,
                }),
            "gemini" | "google" => self
                .gemini_provider
                .as_ref()
                .map(|p| p.clone() as Arc<dyn Provider>)
                .ok_or_else(|| ProxyError::ProviderApi {
                    provider: "gemini".to_string(),
                    message: "Gemini provider not configured".to_string(),
                    status_code: None,
                }),
            "ollama" => self
                .ollama_provider
                .as_ref()
                .map(|p| p.clone() as Arc<dyn Provider>)
                .ok_or_else(|| ProxyError::ProviderApi {
                    provider: "ollama".to_string(),
                    message: "Ollama provider not configured".to_string(),
                    status_code: None,
                }),
            _ => Err(ProxyError::ProviderApi {
                provider: provider_name.to_string(),
                message: format!("Unknown provider: {}", provider_name),
                status_code: None,
            }),
        }
    }
}

/// Messages handler
///
/// Handles POST /v1/messages requests with both streaming and non-streaming support.
///
/// # Headers
///
/// - `x-api-key`: API key (validated but not used for auth)
/// - `anthropic-version`: API version (validated)
///
/// # Request Body
///
/// Standard Anthropic Messages API request format.
///
/// # Response
///
/// - Non-streaming: JSON response in Anthropic format
/// - Streaming: Server-Sent Events (SSE) stream
pub async fn messages_handler(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(request): Json<MessagesRequest>,
) -> Result<Response, ProxyError> {
    // Validate required headers
    validate_headers(&headers)?;

    // Log the incoming request
    info!(
        "Incoming request: model={}, max_tokens={}, streaming={}",
        request.model, request.max_tokens, request.stream
    );

    // Resolve model name and determine provider
    let (resolved_model, provider_name) = resolve_model_and_provider(&state, &request.model)?;

    info!(
        "Model mapping: {} → {} (provider: {})",
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
    provider_request.model = resolved_model.clone();

    // Handle streaming vs non-streaming
    if request.stream {
        // TODO: Implement streaming response
        warn!("Streaming not yet implemented, falling back to non-streaming");
        handle_non_streaming(provider, provider_request, &request.model).await
    } else {
        handle_non_streaming(provider, provider_request, &request.model).await
    }
}

/// Validate required headers
fn validate_headers(headers: &HeaderMap) -> ProxyResult<()> {
    // Check for x-api-key header (required by Anthropic API spec)
    if !headers.contains_key("x-api-key") {
        return Err(ProxyError::Conversion {
            message: "Missing required header: x-api-key".to_string(),
        });
    }

    // Check for anthropic-version header (required by Anthropic API spec)
    if !headers.contains_key("anthropic-version") {
        return Err(ProxyError::Conversion {
            message: "Missing required header: anthropic-version".to_string(),
        });
    }

    Ok(())
}

/// Resolve model name and determine which provider to use
fn resolve_model_and_provider(
    state: &AppState,
    model_name: &str,
) -> ProxyResult<(String, String)> {
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

/// Handle non-streaming request
async fn handle_non_streaming(
    provider: Arc<dyn Provider>,
    provider_request: crate::models::providers::ProviderRequest,
    original_model: &str,
) -> Result<Response, ProxyError> {
    // Call provider
    let provider_response = provider.complete(&provider_request).await?;

    // Convert back to Anthropic format
    let anthropic_response = provider_to_anthropic::convert_provider_to_anthropic(
        &provider_response,
        original_model,
    )?;

    // Return JSON response
    Ok((StatusCode::OK, Json(anthropic_response)).into_response())
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

    #[test]
    fn test_app_state_creation() {
        let config = create_test_config();
        let state = AppState::new(config).unwrap();

        assert!(state.openai_provider.is_some());
        assert!(state.gemini_provider.is_some());
        assert!(state.ollama_provider.is_some());
    }

    #[test]
    fn test_app_state_empty_api_keys() {
        let mut config = create_test_config();
        config.providers.openai.api_key = "".to_string();
        config.providers.google.api_key = "".to_string();

        let state = AppState::new(config).unwrap();

        assert!(state.openai_provider.is_none());
        assert!(state.gemini_provider.is_none());
        assert!(state.ollama_provider.is_some()); // Ollama doesn't require API key
    }

    #[test]
    fn test_get_provider_openai() {
        let config = create_test_config();
        let state = AppState::new(config).unwrap();

        let provider = state.get_provider("openai").unwrap();
        assert_eq!(provider.name(), "openai");
    }

    #[test]
    fn test_get_provider_gemini() {
        let config = create_test_config();
        let state = AppState::new(config).unwrap();

        let provider = state.get_provider("gemini").unwrap();
        assert_eq!(provider.name(), "gemini");
    }

    #[test]
    fn test_get_provider_ollama() {
        let config = create_test_config();
        let state = AppState::new(config).unwrap();

        let provider = state.get_provider("ollama").unwrap();
        assert_eq!(provider.name(), "ollama");
    }

    #[test]
    fn test_get_provider_unknown() {
        let config = create_test_config();
        let state = AppState::new(config).unwrap();

        let result = state.get_provider("unknown");
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_headers_success() {
        let mut headers = HeaderMap::new();
        headers.insert("x-api-key", "test-key".parse().unwrap());
        headers.insert("anthropic-version", "2023-06-01".parse().unwrap());

        let result = validate_headers(&headers);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_headers_missing_api_key() {
        let mut headers = HeaderMap::new();
        headers.insert("anthropic-version", "2023-06-01".parse().unwrap());

        let result = validate_headers(&headers);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_headers_missing_version() {
        let mut headers = HeaderMap::new();
        headers.insert("x-api-key", "test-key".parse().unwrap());

        let result = validate_headers(&headers);
        assert!(result.is_err());
    }

    #[test]
    fn test_resolve_model_and_provider_haiku() {
        let config = create_test_config();
        let state = AppState::new(config).unwrap();

        let (model, provider) = resolve_model_and_provider(&state, "haiku").unwrap();
        assert_eq!(model, "gpt-4o-mini");
        assert_eq!(provider, "openai");
    }

    #[test]
    fn test_resolve_model_and_provider_sonnet() {
        let config = create_test_config();
        let state = AppState::new(config).unwrap();

        let (model, provider) = resolve_model_and_provider(&state, "sonnet").unwrap();
        assert_eq!(model, "gpt-4o");
        assert_eq!(provider, "openai");
    }

    #[test]
    fn test_resolve_model_and_provider_with_prefix() {
        let config = create_test_config();
        let state = AppState::new(config).unwrap();

        let (model, provider) = resolve_model_and_provider(&state, "openai/gpt-4o").unwrap();
        assert_eq!(model, "gpt-4o");
        assert_eq!(provider, "openai");
    }

    #[test]
    fn test_resolve_model_and_provider_ollama() {
        let config = create_test_config();
        let state = AppState::new(config).unwrap();

        let (model, provider) = resolve_model_and_provider(&state, "ollama/llama3").unwrap();
        assert_eq!(model, "llama3");
        assert_eq!(provider, "ollama");
    }

    #[test]
    fn test_resolve_model_and_provider_unknown() {
        let config = create_test_config();
        let state = AppState::new(config).unwrap();

        let result = resolve_model_and_provider(&state, "unknown-model-xyz");
        assert!(result.is_err());
    }
}
