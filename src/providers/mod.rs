//! Provider implementations for different LLM backends
//!
//! This module defines the `Provider` trait and implements it for:
//! - **Ollama**: Local models with OpenAI-compatible API
//! - **OpenAI**: GPT models via OpenAI API
//! - **Gemini**: Google's Gemini models
//!
//! Each provider handles:
//! - Non-streaming completion requests
//! - Streaming completion requests (Server-Sent Events)
//! - Token counting/estimation
//! - Error handling and retries

pub mod gemini;
pub mod ollama;
pub mod openai;

use crate::error::ProxyResult;
use crate::models::providers::{ProviderRequest, ProviderResponse, StreamChunk};
use async_trait::async_trait;
use futures::stream::BoxStream;

/// Provider trait for LLM backends
///
/// All providers must implement this trait to handle completion requests,
/// streaming, and token counting.
#[async_trait]
pub trait Provider: Send + Sync {
    /// Get the provider name (e.g., "ollama", "openai", "gemini")
    fn name(&self) -> &str;

    /// Complete a non-streaming request
    ///
    /// # Arguments
    ///
    /// * `request` - The normalized provider request
    ///
    /// # Errors
    ///
    /// Returns error if the API call fails or response cannot be parsed
    async fn complete(&self, request: &ProviderRequest) -> ProxyResult<ProviderResponse>;

    /// Complete a streaming request
    ///
    /// Returns a stream of chunks that will be sent as Server-Sent Events.
    ///
    /// # Arguments
    ///
    /// * `request` - The normalized provider request
    ///
    /// # Errors
    ///
    /// Returns error if the stream cannot be established
    async fn complete_stream(
        &self,
        request: &ProviderRequest,
    ) -> ProxyResult<BoxStream<'static, ProxyResult<StreamChunk>>>;

    /// Count or estimate tokens in the request
    ///
    /// Different providers use different methods:
    /// - OpenAI: Uses tiktoken for accurate counting
    /// - Others: Estimate based on word count
    ///
    /// # Arguments
    ///
    /// * `request` - The normalized provider request
    ///
    /// # Errors
    ///
    /// Returns error if token counting fails
    async fn count_tokens(&self, request: &ProviderRequest) -> ProxyResult<usize>;
}
