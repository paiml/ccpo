//! Health check endpoint handler
//!
//! Simple health check endpoint that returns service status.
//! Used for monitoring and load balancer health checks.

use axum::{http::StatusCode, response::IntoResponse, Json};
use serde_json::json;

/// Health check handler
///
/// Returns a simple JSON response indicating the service is running.
///
/// # Response
///
/// ```json
/// {
///   "service": "Claude Code Proxy",
///   "status": "ok"
/// }
/// ```
pub async fn health_handler() -> impl IntoResponse {
    (
        StatusCode::OK,
        Json(json!({
            "service": "Claude Code Proxy",
            "status": "ok"
        })),
    )
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_health_endpoint() {
        let response = health_handler().await.into_response();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_health_response_body() {
        let response = health_handler().await.into_response();

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let body_str = String::from_utf8(body.to_vec()).unwrap();

        assert!(body_str.contains("Claude Code Proxy"));
        assert!(body_str.contains("\"status\":\"ok\""));
    }
}
