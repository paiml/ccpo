//! Server setup and routing
//!
//! This module configures the Axum web server with all routes,
//! middleware, and application state.

use crate::handlers::{count_tokens, health, messages};
use axum::{
    routing::{get, post},
    Router,
};
use tower_http::{cors::CorsLayer, trace::TraceLayer};

/// Create the Axum router with all routes and middleware
///
/// # Routes
///
/// - `GET /` - Health check
/// - `POST /v1/messages` - Main completion endpoint
/// - `POST /v1/messages/count_tokens` - Token counting
///
/// # Middleware
///
/// - CORS: Permissive CORS for development
/// - Tracing: HTTP request/response logging
pub fn create_router(state: messages::AppState) -> Router {
    Router::new()
        // Health check endpoint
        .route("/", get(health::health_handler))
        // Messages API endpoints
        .route("/v1/messages", post(messages::messages_handler))
        .route(
            "/v1/messages/count_tokens",
            post(count_tokens::count_tokens_handler),
        )
        // Middleware
        .layer(TraceLayer::new_for_http())
        .layer(CorsLayer::permissive())
        // Application state
        .with_state(state)
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
    use axum::{
        body::Body,
        http::{Request, StatusCode},
    };
    use tower::ServiceExt; // for `oneshot`

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

    #[tokio::test]
    async fn test_health_route() {
        let config = create_test_config();
        let state = messages::AppState::new(config).unwrap();
        let app = create_router(state);

        let request = Request::builder()
            .uri("/")
            .method("GET")
            .body(Body::empty())
            .unwrap();

        let response = app.oneshot(request).await.unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_messages_route_exists() {
        let config = create_test_config();
        let state = messages::AppState::new(config).unwrap();
        let app = create_router(state);

        // This should fail with missing headers, not 404
        let request = Request::builder()
            .uri("/v1/messages")
            .method("POST")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"model":"haiku","max_tokens":100,"messages":[]}"#))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();

        // Should not be 404 (route exists)
        assert_ne!(response.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_count_tokens_route_exists() {
        let config = create_test_config();
        let state = messages::AppState::new(config).unwrap();
        let app = create_router(state);

        // Basic request to verify route exists
        let request = Request::builder()
            .uri("/v1/messages/count_tokens")
            .method("POST")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"model":"haiku","max_tokens":100,"messages":[]}"#))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();

        // Should not be 404 (route exists)
        assert_ne!(response.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_404_for_unknown_route() {
        let config = create_test_config();
        let state = messages::AppState::new(config).unwrap();
        let app = create_router(state);

        let request = Request::builder()
            .uri("/unknown")
            .method("GET")
            .body(Body::empty())
            .unwrap();

        let response = app.oneshot(request).await.unwrap();

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }
}
