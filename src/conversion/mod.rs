//! Format conversion between Anthropic API and provider APIs
//!
//! This module handles the translation of requests and responses between
//! the Anthropic Claude API format and the various provider formats (OpenAI,
//! Gemini, Ollama).
//!
//! # Architecture
//!
//! The conversion layer uses intermediate `ProviderRequest` and `ProviderResponse`
//! types to normalize the different provider formats. This allows us to:
//!
//! 1. Convert from Anthropic format to the internal provider format
//! 2. Convert from the internal provider format to the specific provider's API format
//! 3. Convert from the provider's response format back to Anthropic format
//!
//! # Key Challenges
//!
//! - **System messages**: Anthropic has a separate `system` field, providers use messages
//! - **Tool results**: OpenAI requires tool results in a specific format
//! - **Content blocks**: Anthropic uses content blocks, providers often use strings
//! - **Gemini schema**: Gemini has strict schema requirements, needs cleaning
//! - **Token limits**: Different providers have different max_tokens limits

pub mod anthropic_to_provider;
pub mod provider_to_anthropic;

pub use anthropic_to_provider::convert_anthropic_to_provider;
pub use provider_to_anthropic::convert_provider_to_anthropic;
