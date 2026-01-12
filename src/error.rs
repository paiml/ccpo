//! Error types for ccpo
//!
//! This module defines all error types used throughout the application,
//! with proper HTTP status code mappings for API responses.
//!
//! # Example
//!
//! ```
//! use ccpo::error::{ProxyError, ProxyResult};
//!
//! fn validate_model(model: &str) -> ProxyResult<()> {
//!     if model.is_empty() {
//!         return Err(ProxyError::InvalidInput {
//!             field: "model".to_string(),
//!             message: "Model name cannot be empty".to_string(),
//!         });
//!     }
//!     Ok(())
//! }
//! ```

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;
use thiserror::Error;

/// Result type alias for proxy operations
pub type ProxyResult<T> = Result<T, ProxyError>;

/// Main error type for the proxy application
#[derive(Error, Debug)]
pub enum ProxyError {
    /// Configuration error
    #[error("Configuration error: {0}")]
    Config(#[from] config::ConfigError),

    /// Invalid input from user
    #[error("Invalid input for field '{field}': {message}")]
    InvalidInput { field: String, message: String },

    /// Model not found
    #[error("Model '{model}' not found. Valid models for provider '{provider}': {valid_models}")]
    ModelNotFound {
        model: String,
        provider: String,
        valid_models: String,
    },

    /// Missing required header
    #[error("Missing required header: {header}")]
    MissingHeader { header: String },

    /// Invalid API key
    #[error("Invalid or missing API key")]
    InvalidApiKey,

    /// Provider API error
    #[error("Provider '{provider}' API error: {message}")]
    ProviderApi {
        provider: String,
        message: String,
        status_code: Option<u16>,
    },

    /// Provider timeout
    #[error("Provider '{provider}' request timed out after {timeout_secs}s")]
    ProviderTimeout {
        provider: String,
        timeout_secs: u64,
    },

    /// Rate limit exceeded
    #[error("Rate limit exceeded for provider '{provider}'. Retry after {retry_after_secs}s")]
    RateLimitExceeded {
        provider: String,
        retry_after_secs: u64,
    },

    /// Conversion error (format translation)
    #[error("Conversion error: {message}")]
    Conversion { message: String },

    /// Streaming error
    #[error("Streaming error: {message}")]
    Streaming { message: String },

    /// JSON serialization/deserialization error
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// HTTP request error
    #[error("HTTP request error: {0}")]
    Http(String),

    /// Internal server error
    #[error("Internal server error: {message}")]
    Internal { message: String },
}

impl ProxyError {
    /// Get the HTTP status code for this error
    pub fn status_code(&self) -> StatusCode {
        match self {
            // 400 Bad Request
            ProxyError::Config(_) => StatusCode::BAD_REQUEST,
            ProxyError::InvalidInput { .. } => StatusCode::BAD_REQUEST,
            ProxyError::Json(_) => StatusCode::BAD_REQUEST,
            ProxyError::Conversion { .. } => StatusCode::BAD_REQUEST,

            // 401 Unauthorized
            ProxyError::InvalidApiKey => StatusCode::UNAUTHORIZED,

            // 404 Not Found
            ProxyError::ModelNotFound { .. } => StatusCode::NOT_FOUND,

            // 422 Unprocessable Entity
            ProxyError::MissingHeader { .. } => StatusCode::UNPROCESSABLE_ENTITY,

            // 429 Too Many Requests
            ProxyError::RateLimitExceeded { .. } => StatusCode::TOO_MANY_REQUESTS,

            // 502 Bad Gateway (upstream provider error)
            ProxyError::ProviderApi { .. } => StatusCode::BAD_GATEWAY,

            // 503 Service Unavailable
            ProxyError::ProviderTimeout { .. } => StatusCode::SERVICE_UNAVAILABLE,

            // 500 Internal Server Error
            ProxyError::Streaming { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ProxyError::Http(_) => StatusCode::INTERNAL_SERVER_ERROR,
            ProxyError::Internal { .. } => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    /// Get the error type string for API responses
    pub fn error_type(&self) -> &'static str {
        match self {
            ProxyError::Config(_) => "configuration_error",
            ProxyError::InvalidInput { .. } => "invalid_request_error",
            ProxyError::ModelNotFound { .. } => "not_found_error",
            ProxyError::MissingHeader { .. } => "invalid_request_error",
            ProxyError::InvalidApiKey => "authentication_error",
            ProxyError::ProviderApi { .. } => "api_error",
            ProxyError::ProviderTimeout { .. } => "timeout_error",
            ProxyError::RateLimitExceeded { .. } => "rate_limit_error",
            ProxyError::Conversion { .. } => "invalid_request_error",
            ProxyError::Streaming { .. } => "api_error",
            ProxyError::Json(_) => "invalid_request_error",
            ProxyError::Http(_) => "api_error",
            ProxyError::Internal { .. } => "internal_server_error",
        }
    }

    /// Check if this error is retryable
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            ProxyError::ProviderTimeout { .. }
                | ProxyError::RateLimitExceeded { .. }
                | ProxyError::Http(_)
                | ProxyError::ProviderApi {
                    status_code: Some(500..=599),
                    ..
                }
        )
    }

    /// Get retry delay in seconds (if retryable)
    pub fn retry_delay_secs(&self) -> Option<u64> {
        match self {
            ProxyError::RateLimitExceeded {
                retry_after_secs, ..
            } => Some(*retry_after_secs),
            ProxyError::ProviderTimeout { .. } => Some(5), // Wait 5s before retry
            ProxyError::Http(_) => Some(2),                // Wait 2s before retry
            ProxyError::ProviderApi {
                status_code: Some(500..=599),
                ..
            } => Some(3), // Wait 3s before retry
            _ => None,
        }
    }
}

/// Convert ProxyError into an Axum HTTP response
impl IntoResponse for ProxyError {
    fn into_response(self) -> Response {
        let status = self.status_code();
        let error_type = self.error_type();
        let message = self.to_string();

        let body = json!({
            "type": "error",
            "error": {
                "type": error_type,
                "message": message,
            }
        });

        (status, Json(body)).into_response()
    }
}

/// Convert from reqwest errors
impl From<reqwest::Error> for ProxyError {
    fn from(err: reqwest::Error) -> Self {
        if err.is_timeout() {
            ProxyError::ProviderTimeout {
                provider: "unknown".to_string(),
                timeout_secs: 30,
            }
        } else if err.is_status() {
            let status = err.status().map(|s| s.as_u16());
            ProxyError::ProviderApi {
                provider: "unknown".to_string(),
                message: err.to_string(),
                status_code: status,
            }
        } else {
            ProxyError::Http(err.to_string())
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_error() {
        let err = ProxyError::Config(config::ConfigError::Message(
            "Invalid config".to_string(),
        ));
        assert_eq!(err.status_code(), StatusCode::BAD_REQUEST);
        assert_eq!(err.error_type(), "configuration_error");
        assert!(!err.is_retryable());
        assert!(err.to_string().contains("Configuration error"));
    }

    #[test]
    fn test_invalid_input_error() {
        let err = ProxyError::InvalidInput {
            field: "max_tokens".to_string(),
            message: "Must be positive".to_string(),
        };
        assert_eq!(err.status_code(), StatusCode::BAD_REQUEST);
        assert_eq!(err.error_type(), "invalid_request_error");
        assert!(!err.is_retryable());
        assert!(err.to_string().contains("max_tokens"));
        assert!(err.to_string().contains("Must be positive"));
    }

    #[test]
    fn test_model_not_found_error() {
        let err = ProxyError::ModelNotFound {
            model: "unknown-model".to_string(),
            provider: "openai".to_string(),
            valid_models: "gpt-4o, gpt-4o-mini".to_string(),
        };
        assert_eq!(err.status_code(), StatusCode::NOT_FOUND);
        assert_eq!(err.error_type(), "not_found_error");
        assert!(!err.is_retryable());
        assert!(err.to_string().contains("unknown-model"));
        assert!(err.to_string().contains("gpt-4o"));
    }

    #[test]
    fn test_missing_header_error() {
        let err = ProxyError::MissingHeader {
            header: "x-api-key".to_string(),
        };
        assert_eq!(err.status_code(), StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(err.error_type(), "invalid_request_error");
        assert!(!err.is_retryable());
        assert!(err.to_string().contains("x-api-key"));
    }

    #[test]
    fn test_invalid_api_key_error() {
        let err = ProxyError::InvalidApiKey;
        assert_eq!(err.status_code(), StatusCode::UNAUTHORIZED);
        assert_eq!(err.error_type(), "authentication_error");
        assert!(!err.is_retryable());
        assert!(err.to_string().contains("Invalid or missing API key"));
    }

    #[test]
    fn test_provider_api_error() {
        let err = ProxyError::ProviderApi {
            provider: "openai".to_string(),
            message: "Invalid request".to_string(),
            status_code: Some(400),
        };
        assert_eq!(err.status_code(), StatusCode::BAD_GATEWAY);
        assert_eq!(err.error_type(), "api_error");
        assert!(!err.is_retryable()); // 4xx not retryable
        assert!(err.to_string().contains("openai"));
        assert!(err.to_string().contains("Invalid request"));
    }

    #[test]
    fn test_provider_api_error_retryable() {
        let err = ProxyError::ProviderApi {
            provider: "gemini".to_string(),
            message: "Internal error".to_string(),
            status_code: Some(503),
        };
        assert_eq!(err.status_code(), StatusCode::BAD_GATEWAY);
        assert!(err.is_retryable()); // 5xx is retryable
        assert_eq!(err.retry_delay_secs(), Some(3));
    }

    #[test]
    fn test_provider_timeout_error() {
        let err = ProxyError::ProviderTimeout {
            provider: "ollama".to_string(),
            timeout_secs: 60,
        };
        assert_eq!(err.status_code(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(err.error_type(), "timeout_error");
        assert!(err.is_retryable());
        assert_eq!(err.retry_delay_secs(), Some(5));
        assert!(err.to_string().contains("60s"));
    }

    #[test]
    fn test_rate_limit_error() {
        let err = ProxyError::RateLimitExceeded {
            provider: "openai".to_string(),
            retry_after_secs: 30,
        };
        assert_eq!(err.status_code(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(err.error_type(), "rate_limit_error");
        assert!(err.is_retryable());
        assert_eq!(err.retry_delay_secs(), Some(30));
        assert!(err.to_string().contains("30s"));
    }

    #[test]
    fn test_conversion_error() {
        let err = ProxyError::Conversion {
            message: "Failed to convert tool schema".to_string(),
        };
        assert_eq!(err.status_code(), StatusCode::BAD_REQUEST);
        assert_eq!(err.error_type(), "invalid_request_error");
        assert!(!err.is_retryable());
        assert!(err.to_string().contains("tool schema"));
    }

    #[test]
    fn test_streaming_error() {
        let err = ProxyError::Streaming {
            message: "SSE connection lost".to_string(),
        };
        assert_eq!(err.status_code(), StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(err.error_type(), "api_error");
        assert!(!err.is_retryable());
        assert!(err.to_string().contains("SSE"));
    }

    #[test]
    fn test_json_error() {
        let json_err = serde_json::from_str::<serde_json::Value>("invalid json")
            .unwrap_err();
        let err = ProxyError::from(json_err);
        assert_eq!(err.status_code(), StatusCode::BAD_REQUEST);
        assert_eq!(err.error_type(), "invalid_request_error");
        assert!(!err.is_retryable());
    }

    #[test]
    fn test_http_error() {
        let err = ProxyError::Http("Connection refused".to_string());
        assert_eq!(err.status_code(), StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(err.error_type(), "api_error");
        assert!(err.is_retryable());
        assert_eq!(err.retry_delay_secs(), Some(2));
    }

    #[test]
    fn test_internal_error() {
        let err = ProxyError::Internal {
            message: "Unexpected state".to_string(),
        };
        assert_eq!(err.status_code(), StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(err.error_type(), "internal_server_error");
        assert!(!err.is_retryable());
        assert!(err.to_string().contains("Unexpected state"));
    }

    #[test]
    fn test_into_response() {
        let err = ProxyError::ModelNotFound {
            model: "gpt-5".to_string(),
            provider: "openai".to_string(),
            valid_models: "gpt-4o".to_string(),
        };

        let response = err.into_response();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[test]
    fn test_from_reqwest_timeout() {
        // Note: We can't easily create a real reqwest::Error in tests since
        // its constructors are private. We test the timeout error behavior directly.
        let err = ProxyError::ProviderTimeout {
            provider: "test".to_string(),
            timeout_secs: 30,
        };

        assert!(err.is_retryable());
        assert_eq!(err.status_code(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(err.retry_delay_secs(), Some(5));
    }

    #[test]
    fn test_error_type_strings() {
        // Ensure error type strings follow Anthropic API conventions
        assert_eq!(
            ProxyError::InvalidInput {
                field: "test".to_string(),
                message: "test".to_string()
            }
            .error_type(),
            "invalid_request_error"
        );

        assert_eq!(ProxyError::InvalidApiKey.error_type(), "authentication_error");

        assert_eq!(
            ProxyError::ModelNotFound {
                model: "test".to_string(),
                provider: "test".to_string(),
                valid_models: "test".to_string()
            }
            .error_type(),
            "not_found_error"
        );
    }

    #[test]
    fn test_retryable_errors() {
        // Retryable
        assert!(ProxyError::ProviderTimeout {
            provider: "test".to_string(),
            timeout_secs: 30
        }
        .is_retryable());

        assert!(ProxyError::RateLimitExceeded {
            provider: "test".to_string(),
            retry_after_secs: 30
        }
        .is_retryable());

        assert!(ProxyError::Http("test".to_string()).is_retryable());

        assert!(ProxyError::ProviderApi {
            provider: "test".to_string(),
            message: "test".to_string(),
            status_code: Some(503)
        }
        .is_retryable());

        // Not retryable
        assert!(!ProxyError::InvalidInput {
            field: "test".to_string(),
            message: "test".to_string()
        }
        .is_retryable());

        assert!(!ProxyError::ModelNotFound {
            model: "test".to_string(),
            provider: "test".to_string(),
            valid_models: "test".to_string()
        }
        .is_retryable());

        assert!(!ProxyError::InvalidApiKey.is_retryable());

        assert!(!ProxyError::ProviderApi {
            provider: "test".to_string(),
            message: "test".to_string(),
            status_code: Some(400)
        }
        .is_retryable());
    }

    #[test]
    fn test_retry_delays() {
        assert_eq!(
            ProxyError::RateLimitExceeded {
                provider: "test".to_string(),
                retry_after_secs: 45
            }
            .retry_delay_secs(),
            Some(45)
        );

        assert_eq!(
            ProxyError::ProviderTimeout {
                provider: "test".to_string(),
                timeout_secs: 30
            }
            .retry_delay_secs(),
            Some(5)
        );

        assert_eq!(
            ProxyError::Http("test".to_string()).retry_delay_secs(),
            Some(2)
        );

        assert_eq!(
            ProxyError::ProviderApi {
                provider: "test".to_string(),
                message: "test".to_string(),
                status_code: Some(500)
            }
            .retry_delay_secs(),
            Some(3)
        );

        assert_eq!(ProxyError::InvalidApiKey.retry_delay_secs(), None);
    }

    #[test]
    fn test_all_status_codes_mapped() {
        // Ensure all error variants have status codes
        let errors = vec![
            ProxyError::Config(config::ConfigError::Message("test".to_string())),
            ProxyError::InvalidInput {
                field: "test".to_string(),
                message: "test".to_string(),
            },
            ProxyError::ModelNotFound {
                model: "test".to_string(),
                provider: "test".to_string(),
                valid_models: "test".to_string(),
            },
            ProxyError::MissingHeader {
                header: "test".to_string(),
            },
            ProxyError::InvalidApiKey,
            ProxyError::ProviderApi {
                provider: "test".to_string(),
                message: "test".to_string(),
                status_code: None,
            },
            ProxyError::ProviderTimeout {
                provider: "test".to_string(),
                timeout_secs: 30,
            },
            ProxyError::RateLimitExceeded {
                provider: "test".to_string(),
                retry_after_secs: 30,
            },
            ProxyError::Conversion {
                message: "test".to_string(),
            },
            ProxyError::Streaming {
                message: "test".to_string(),
            },
            ProxyError::Json(serde_json::from_str::<serde_json::Value>("invalid").unwrap_err()),
            ProxyError::Http("test".to_string()),
            ProxyError::Internal {
                message: "test".to_string(),
            },
        ];

        for err in errors {
            // Should not panic
            let _ = err.status_code();
            let _ = err.error_type();
        }
    }

    #[test]
    fn test_error_message_formatting() {
        let err = ProxyError::ModelNotFound {
            model: "claude-4".to_string(),
            provider: "anthropic".to_string(),
            valid_models: "claude-3-opus, claude-3-sonnet, claude-3-haiku".to_string(),
        };

        let message = err.to_string();
        assert!(message.contains("claude-4"));
        assert!(message.contains("anthropic"));
        assert!(message.contains("claude-3-opus"));
    }

    #[test]
    fn test_result_type_alias() {
        fn returns_result() -> ProxyResult<String> {
            Ok("success".to_string())
        }

        fn returns_error() -> ProxyResult<String> {
            Err(ProxyError::InvalidInput {
                field: "test".to_string(),
                message: "error".to_string(),
            })
        }

        assert!(returns_result().is_ok());
        assert!(returns_error().is_err());
    }
}
