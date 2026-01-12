//! Claude Code Proxy for Other APIs (ccpo)
//!
//! A high-performance proxy server that translates Anthropic Claude API requests
//! to OpenAI, Google Gemini, and Ollama backends.
//!
//! # Architecture
//!
//! The proxy follows a layered architecture:
//! - **Models**: Data types for Anthropic API and internal provider formats
//! - **Providers**: Backend implementations (OpenAI, Gemini, Ollama)
//! - **Conversion**: Format translation between Anthropic and provider APIs
//! - **Handlers**: HTTP request handlers
//! - **Server**: Axum web server setup

pub mod config;
pub mod conversion;
pub mod error;
pub mod handlers;
pub mod logging;
pub mod mapping;
pub mod models;
pub mod providers;
pub mod server;

// Modules will be added as we implement them:
// pub mod streaming;
// pub mod handlers;
// pub mod server;
