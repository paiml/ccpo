//! HTTP request handlers
//!
//! This module contains all HTTP endpoint handlers for the proxy server:
//! - `health`: GET / - Health check endpoint
//! - `messages`: POST /v1/messages - Main completion endpoint
//! - `count_tokens`: POST /v1/messages/count_tokens - Token counting endpoint

pub mod health;
pub mod messages;
pub mod count_tokens;
