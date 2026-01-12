/// Data models for API requests and responses
///
/// This module contains the type definitions for:
/// - Anthropic API format (client-facing)
/// - Internal provider format (backend abstraction)
/// - Conversion between formats

pub mod anthropic;
pub mod providers;

// Re-export commonly used types
pub use anthropic::{
    ContentBlock, Message, MessagesRequest, MessagesResponse, StreamEvent, Tool, ToolChoice,
    Usage,
};
pub use providers::{ProviderRequest, ProviderResponse, StreamChunk};
