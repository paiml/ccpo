//! Logging configuration and setup
//!
//! This module provides centralized logging configuration using the `tracing` crate.
//! Features include:
//! - Colored console output for better readability
//! - Configurable log levels (trace, debug, info, warn, error)
//! - Filtering to suppress verbose third-party logs
//! - Request/response logging with model mapping details
//! - JSON formatting option for production environments

use std::io;
use tracing::Level;
use tracing_subscriber::{
    fmt::{self, format::FmtSpan},
    layer::SubscriberExt,
    util::SubscriberInitExt,
    EnvFilter, Layer,
};

/// Initialize the tracing subscriber with the given configuration
///
/// This sets up the global tracing subscriber with colored output (if enabled),
/// proper log level filtering, and suppression of verbose third-party logs.
///
/// # Arguments
///
/// * `level` - Log level string ("trace", "debug", "info", "warn", "error")
/// * `format` - Log format ("pretty" or "json")
/// * `suppress_provider_logs` - Whether to suppress verbose third-party logs
///
/// # Errors
///
/// Returns an error if the subscriber has already been initialized or if
/// the configuration is invalid.
///
/// # Example
///
/// ```no_run
/// use ccpo::logging::init_logging;
///
/// init_logging("info", "pretty", true).expect("Failed to initialize logging");
/// ```
pub fn init_logging(
    level: &str,
    format: &str,
    suppress_provider_logs: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    // Parse the log level
    let level = parse_log_level(level)?;

    // Build the environment filter
    let env_filter = build_env_filter(level, suppress_provider_logs);

    // Build the formatting layer based on configuration
    let is_json = format.to_lowercase() == "json";
    if is_json {
        // JSON formatting for production/structured logging
        let json_layer = fmt::layer()
            .json()
            .with_span_events(FmtSpan::CLOSE)
            .with_filter(env_filter);

        tracing_subscriber::registry().with(json_layer).init();
    } else {
        // Pretty colored formatting for development
        let fmt_layer = fmt::layer()
            .with_ansi(true) // Enable colored output
            .with_target(true) // Show the target (module path)
            .with_line_number(true) // Show line numbers
            .with_thread_ids(false) // Don't clutter with thread IDs
            .with_span_events(FmtSpan::CLOSE) // Log when spans close
            .with_writer(io::stderr) // Write to stderr
            .with_filter(env_filter);

        tracing_subscriber::registry().with(fmt_layer).init();
    }

    tracing::info!(
        log_level = level.as_str(),
        format = format,
        suppress_provider_logs = suppress_provider_logs,
        "Logging initialized"
    );

    Ok(())
}

/// Parse a log level string into a `tracing::Level`
///
/// # Errors
///
/// Returns an error if the log level string is not recognized
fn parse_log_level(level: &str) -> Result<Level, String> {
    match level.to_lowercase().as_str() {
        "trace" => Ok(Level::TRACE),
        "debug" => Ok(Level::DEBUG),
        "info" => Ok(Level::INFO),
        "warn" | "warning" => Ok(Level::WARN),
        "error" => Ok(Level::ERROR),
        _ => Err(format!("Invalid log level: {}", level)),
    }
}

/// Build an environment filter with proper log level and third-party filtering
///
/// This function creates an `EnvFilter` that:
/// 1. Sets the default log level for our crate
/// 2. Suppresses verbose logs from third-party crates (if enabled)
/// 3. Respects the RUST_LOG environment variable if set
fn build_env_filter(level: Level, suppress_provider_logs: bool) -> EnvFilter {
    // Start with the default level for our crate
    let default_directive = format!("ccpo={}", level);

    // Build the filter starting with our default
    let mut filter = EnvFilter::try_from_default_env()
        .or_else(|_| EnvFilter::try_new(&default_directive))
        .unwrap_or_else(|_| EnvFilter::new("info"));

    // Suppress verbose third-party logs if requested
    // This is similar to the Python implementation's LiteLLMFilter
    if suppress_provider_logs {
        filter = filter
            // Suppress verbose HTTP client logs
            .add_directive("hyper=warn".parse().unwrap())
            .add_directive("reqwest=warn".parse().unwrap())
            // Suppress verbose tokio logs
            .add_directive("tokio=warn".parse().unwrap())
            .add_directive("runtime=warn".parse().unwrap())
            // Suppress verbose tower/axum logs
            .add_directive("tower=warn".parse().unwrap())
            .add_directive("tower_http=warn".parse().unwrap())
            .add_directive("axum=info".parse().unwrap())
            // Suppress other noisy crates
            .add_directive("h2=warn".parse().unwrap())
            .add_directive("rustls=warn".parse().unwrap())
            .add_directive("mio=warn".parse().unwrap());
    }

    filter
}

/// Log a request with model mapping information
///
/// This is a helper function to log incoming requests with details about
/// the model mapping (e.g., "Claude haiku → openai/o3-mini")
#[inline]
pub fn log_request(
    method: &str,
    path: &str,
    original_model: &str,
    mapped_model: Option<&str>,
) {
    if let Some(mapped) = mapped_model {
        tracing::info!(
            method = method,
            path = path,
            original_model = original_model,
            mapped_model = mapped,
            "Request received: {} → {}",
            original_model,
            mapped
        );
    } else {
        tracing::info!(
            method = method,
            path = path,
            model = original_model,
            "Request received",
        );
    }
}

/// Log a response with timing information
///
/// This is a helper function to log outgoing responses with status code
/// and request duration
#[inline]
pub fn log_response(
    method: &str,
    path: &str,
    status: u16,
    duration_ms: u64,
) {
    tracing::info!(
        method = method,
        path = path,
        status = status,
        duration_ms = duration_ms,
        "Response sent ({} ms)",
        duration_ms
    );
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_log_level_valid() {
        assert_eq!(parse_log_level("trace").unwrap(), Level::TRACE);
        assert_eq!(parse_log_level("debug").unwrap(), Level::DEBUG);
        assert_eq!(parse_log_level("info").unwrap(), Level::INFO);
        assert_eq!(parse_log_level("warn").unwrap(), Level::WARN);
        assert_eq!(parse_log_level("warning").unwrap(), Level::WARN);
        assert_eq!(parse_log_level("error").unwrap(), Level::ERROR);
    }

    #[test]
    fn test_parse_log_level_case_insensitive() {
        assert_eq!(parse_log_level("TRACE").unwrap(), Level::TRACE);
        assert_eq!(parse_log_level("Info").unwrap(), Level::INFO);
        assert_eq!(parse_log_level("ERROR").unwrap(), Level::ERROR);
    }

    #[test]
    fn test_parse_log_level_invalid() {
        assert!(parse_log_level("invalid").is_err());
        assert!(parse_log_level("").is_err());
        assert!(parse_log_level("fatal").is_err());
    }

    #[test]
    fn test_build_env_filter_with_suppression() {
        let filter = build_env_filter(Level::INFO, true);
        // We can't easily test the internal directives, but we can verify it builds
        assert!(format!("{:?}", filter).contains("EnvFilter"));
    }

    #[test]
    fn test_build_env_filter_without_suppression() {
        let filter = build_env_filter(Level::DEBUG, false);
        assert!(format!("{:?}", filter).contains("EnvFilter"));
    }

    #[test]
    fn test_build_env_filter_different_levels() {
        let trace_filter = build_env_filter(Level::TRACE, false);
        let error_filter = build_env_filter(Level::ERROR, false);

        assert!(format!("{:?}", trace_filter).contains("EnvFilter"));
        assert!(format!("{:?}", error_filter).contains("EnvFilter"));
    }

    // Note: We can't easily test init_logging() in unit tests because
    // tracing_subscriber can only be initialized once per process.
    // These would be better tested in integration tests.

    #[test]
    fn test_log_request_with_mapping() {
        // This test just ensures the function compiles and doesn't panic
        log_request(
            "POST",
            "/v1/messages",
            "claude-3-haiku-20240307",
            Some("openai/o3-mini"),
        );
    }

    #[test]
    fn test_log_request_without_mapping() {
        log_request(
            "POST",
            "/v1/messages",
            "gpt-4",
            None,
        );
    }

    #[test]
    fn test_log_response() {
        log_response("POST", "/v1/messages", 200, 150);
        log_response("GET", "/", 200, 5);
        log_response("POST", "/v1/messages", 500, 3000);
    }
}
