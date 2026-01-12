//! Model name mapping and validation
//!
//! This module handles the translation of model names from the Anthropic API format
//! to provider-specific formats. It implements the following logic:
//!
//! 1. **Strip provider prefixes**: Remove `anthropic/`, `openai/`, `gemini/`, `ollama/`
//! 2. **Map Claude aliases**: Translate `haiku`, `sonnet`, `opus` to configured models
//! 3. **Add provider prefixes**: Prefix models with their provider (e.g., `openai/gpt-4`)
//! 4. **Validate**: Ensure the model exists in the provider's supported list
//!
//! # Example
//!
//! ```no_run
//! use ccpo::config::{ModelsConfig, ModelList};
//! use ccpo::mapping::ModelMapper;
//!
//! let models_config = ModelsConfig {
//!     big: "o3-mini".to_string(),
//!     small: "o3-mini".to_string(),
//!     openai_list: ModelList {
//!         supported: vec!["o3-mini".to_string(), "gpt-4o".to_string()],
//!     },
//!     gemini_list: ModelList {
//!         supported: vec!["gemini-2.0-flash".to_string()],
//!     },
//!     ollama_list: ModelList {
//!         supported: vec!["llama3".to_string()],
//!     },
//! };
//!
//! let mapper = ModelMapper::new(models_config);
//!
//! // Map Claude alias to configured model
//! let result = mapper.resolve_model("haiku", "openai");
//! assert_eq!(result.unwrap(), "openai/o3-mini");
//!
//! // Strip and re-add prefix
//! let result = mapper.resolve_model("openai/gpt-4o", "openai");
//! assert_eq!(result.unwrap(), "openai/gpt-4o");
//! ```

use crate::config::ModelsConfig;
use crate::error::{ProxyError, ProxyResult};
use std::collections::HashSet;

/// Model mapper that handles model name resolution and validation
pub struct ModelMapper {
    /// Configuration for model mappings
    config: ModelsConfig,

    /// Set of OpenAI models for fast lookup
    openai_models: HashSet<String>,

    /// Set of Gemini models for fast lookup
    gemini_models: HashSet<String>,

    /// Set of Ollama models for fast lookup
    ollama_models: HashSet<String>,
}

impl ModelMapper {
    /// Create a new model mapper with the given configuration
    pub fn new(config: ModelsConfig) -> Self {
        let openai_models = config.openai_list.supported.iter().cloned().collect();
        let gemini_models = config.gemini_list.supported.iter().cloned().collect();
        let ollama_models = config.ollama_list.supported.iter().cloned().collect();

        Self {
            config,
            openai_models,
            gemini_models,
            ollama_models,
        }
    }

    /// Resolve a model name to a provider-prefixed model
    ///
    /// This method:
    /// 1. Strips any existing provider prefix
    /// 2. Maps Claude aliases (haiku/sonnet/opus) to configured models
    /// 3. Determines the provider (from prefix or model list lookup)
    /// 4. Validates the model exists in the provider's list
    /// 5. Returns the fully-qualified provider/model string
    ///
    /// # Arguments
    ///
    /// * `model` - The model name from the request (may have prefix)
    /// * `preferred_provider` - The default provider to use if not specified
    ///
    /// # Errors
    ///
    /// Returns an error if the model is not found in any provider's list
    pub fn resolve_model(&self, model: &str, preferred_provider: &str) -> ProxyResult<String> {
        // Step 1: Strip provider prefix if present
        let (original_provider, base_model) = self.strip_provider_prefix(model);

        // Step 2: Map Claude aliases to configured models
        let mapped_model = self.map_claude_alias(&base_model);

        // Step 3: Determine which provider to use
        let provider = if let Some(prov) = original_provider {
            // Use the explicitly specified provider
            prov
        } else {
            // Look up which provider has this model
            self.find_provider_for_model(&mapped_model)
                .unwrap_or(preferred_provider)
        };

        // Step 4: Validate the model exists in the provider's list
        self.validate_model(&mapped_model, provider)?;

        // Step 5: Return fully-qualified model name
        Ok(format!("{}/{}", provider, mapped_model))
    }

    /// Strip provider prefix from model name
    ///
    /// Returns (Option<provider>, base_model)
    fn strip_provider_prefix<'a>(&self, model: &'a str) -> (Option<&'a str>, String) {
        let prefixes = ["anthropic/", "openai/", "gemini/", "ollama/"];

        for prefix in &prefixes {
            if model.starts_with(prefix) {
                let provider = prefix.trim_end_matches('/');
                let base = model.strip_prefix(prefix).unwrap().to_string();
                return (Some(provider), base);
            }
        }

        (None, model.to_string())
    }

    /// Map Claude model aliases to configured models
    ///
    /// - `haiku` → `small_model`
    /// - `sonnet` → `big_model`
    /// - `opus` → `big_model`
    /// - Anything starting with `claude-` and containing `haiku` → `small_model`
    /// - Anything starting with `claude-` and containing `sonnet` → `big_model`
    /// - Anything starting with `claude-` and containing `opus` → `big_model`
    fn map_claude_alias(&self, model: &str) -> String {
        let lower = model.to_lowercase();

        // Exact alias matches
        match lower.as_str() {
            "haiku" => return self.config.small.clone(),
            "sonnet" => return self.config.big.clone(),
            "opus" => return self.config.big.clone(),
            _ => {}
        }

        // Claude model family detection
        if lower.starts_with("claude-") {
            if lower.contains("haiku") {
                return self.config.small.clone();
            } else if lower.contains("sonnet") {
                return self.config.big.clone();
            } else if lower.contains("opus") {
                return self.config.big.clone();
            }
        }

        // No mapping needed, return as-is
        model.to_string()
    }

    /// Find which provider has this model
    ///
    /// Returns the provider name if found in any provider's list
    fn find_provider_for_model(&self, model: &str) -> Option<&str> {
        if self.openai_models.contains(model) {
            Some("openai")
        } else if self.gemini_models.contains(model) {
            Some("gemini")
        } else if self.ollama_models.contains(model) {
            Some("ollama")
        } else {
            None
        }
    }

    /// Validate that a model exists in the provider's list
    ///
    /// # Errors
    ///
    /// Returns `ModelNotFound` error if the model is not in the provider's list
    fn validate_model(&self, model: &str, provider: &str) -> ProxyResult<()> {
        let model_list = match provider {
            "openai" => &self.openai_models,
            "gemini" | "google" => &self.gemini_models,
            "ollama" => &self.ollama_models,
            _ => {
                return Err(ProxyError::ModelNotFound {
                    model: model.to_string(),
                    provider: provider.to_string(),
                    valid_models: "unknown provider".to_string(),
                });
            }
        };

        if !model_list.contains(model) {
            let valid_models = model_list
                .iter()
                .cloned()
                .collect::<Vec<_>>()
                .join(", ");

            return Err(ProxyError::ModelNotFound {
                model: model.to_string(),
                provider: provider.to_string(),
                valid_models,
            });
        }

        Ok(())
    }

    /// Get the list of all supported models across all providers
    pub fn all_models(&self) -> Vec<String> {
        let mut models = Vec::new();

        for model in &self.openai_models {
            models.push(format!("openai/{}", model));
        }
        for model in &self.gemini_models {
            models.push(format!("gemini/{}", model));
        }
        for model in &self.ollama_models {
            models.push(format!("ollama/{}", model));
        }

        models
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ModelList;

    fn create_test_config() -> ModelsConfig {
        ModelsConfig {
            big: "o3-mini".to_string(),
            small: "o3-mini".to_string(),
            openai_list: ModelList {
                supported: vec![
                    "o3-mini".to_string(),
                    "o1".to_string(),
                    "gpt-4o".to_string(),
                    "gpt-4o-mini".to_string(),
                ],
            },
            gemini_list: ModelList {
                supported: vec![
                    "gemini-2.5-pro-preview-03-25".to_string(),
                    "gemini-2.0-flash".to_string(),
                ],
            },
            ollama_list: ModelList {
                supported: vec![
                    "llama3".to_string(),
                    "llama3:8b".to_string(),
                    "mistral".to_string(),
                    "mixtral:8x7b".to_string(),
                ],
            },
        }
    }

    fn create_mapper() -> ModelMapper {
        ModelMapper::new(create_test_config())
    }

    // ========================================================================
    // Prefix Stripping Tests
    // ========================================================================

    #[test]
    fn test_strip_openai_prefix() {
        let mapper = create_mapper();
        let (provider, model) = mapper.strip_provider_prefix("openai/gpt-4o");
        assert_eq!(provider, Some("openai"));
        assert_eq!(model, "gpt-4o");
    }

    #[test]
    fn test_strip_gemini_prefix() {
        let mapper = create_mapper();
        let (provider, model) = mapper.strip_provider_prefix("gemini/gemini-2.0-flash");
        assert_eq!(provider, Some("gemini"));
        assert_eq!(model, "gemini-2.0-flash");
    }

    #[test]
    fn test_strip_ollama_prefix() {
        let mapper = create_mapper();
        let (provider, model) = mapper.strip_provider_prefix("ollama/llama3");
        assert_eq!(provider, Some("ollama"));
        assert_eq!(model, "llama3");
    }

    #[test]
    fn test_strip_anthropic_prefix() {
        let mapper = create_mapper();
        let (provider, model) = mapper.strip_provider_prefix("anthropic/claude-3-sonnet-20240229");
        assert_eq!(provider, Some("anthropic"));
        assert_eq!(model, "claude-3-sonnet-20240229");
    }

    #[test]
    fn test_no_prefix() {
        let mapper = create_mapper();
        let (provider, model) = mapper.strip_provider_prefix("gpt-4o");
        assert_eq!(provider, None);
        assert_eq!(model, "gpt-4o");
    }

    // ========================================================================
    // Claude Alias Mapping Tests
    // ========================================================================

    #[test]
    fn test_map_haiku_alias() {
        let mapper = create_mapper();
        assert_eq!(mapper.map_claude_alias("haiku"), "o3-mini");
        assert_eq!(mapper.map_claude_alias("HAIKU"), "o3-mini");
        assert_eq!(mapper.map_claude_alias("Haiku"), "o3-mini");
    }

    #[test]
    fn test_map_sonnet_alias() {
        let mapper = create_mapper();
        assert_eq!(mapper.map_claude_alias("sonnet"), "o3-mini");
        assert_eq!(mapper.map_claude_alias("SONNET"), "o3-mini");
    }

    #[test]
    fn test_map_opus_alias() {
        let mapper = create_mapper();
        assert_eq!(mapper.map_claude_alias("opus"), "o3-mini");
        assert_eq!(mapper.map_claude_alias("OPUS"), "o3-mini");
    }

    #[test]
    fn test_map_claude_haiku_full_name() {
        let mapper = create_mapper();
        assert_eq!(
            mapper.map_claude_alias("claude-3-haiku-20240307"),
            "o3-mini"
        );
    }

    #[test]
    fn test_map_claude_sonnet_full_name() {
        let mapper = create_mapper();
        assert_eq!(
            mapper.map_claude_alias("claude-3-sonnet-20240229"),
            "o3-mini"
        );
        assert_eq!(
            mapper.map_claude_alias("claude-3-5-sonnet-20240620"),
            "o3-mini"
        );
    }

    #[test]
    fn test_map_claude_opus_full_name() {
        let mapper = create_mapper();
        assert_eq!(
            mapper.map_claude_alias("claude-3-opus-20240229"),
            "o3-mini"
        );
    }

    #[test]
    fn test_no_alias_mapping() {
        let mapper = create_mapper();
        assert_eq!(mapper.map_claude_alias("gpt-4o"), "gpt-4o");
        assert_eq!(mapper.map_claude_alias("llama3"), "llama3");
    }

    // ========================================================================
    // Provider Lookup Tests
    // ========================================================================

    #[test]
    fn test_find_provider_openai() {
        let mapper = create_mapper();
        assert_eq!(mapper.find_provider_for_model("gpt-4o"), Some("openai"));
        assert_eq!(mapper.find_provider_for_model("o3-mini"), Some("openai"));
    }

    #[test]
    fn test_find_provider_gemini() {
        let mapper = create_mapper();
        assert_eq!(
            mapper.find_provider_for_model("gemini-2.0-flash"),
            Some("gemini")
        );
    }

    #[test]
    fn test_find_provider_ollama() {
        let mapper = create_mapper();
        assert_eq!(mapper.find_provider_for_model("llama3"), Some("ollama"));
        assert_eq!(mapper.find_provider_for_model("mistral"), Some("ollama"));
    }

    #[test]
    fn test_find_provider_not_found() {
        let mapper = create_mapper();
        assert_eq!(mapper.find_provider_for_model("unknown-model"), None);
    }

    // ========================================================================
    // Model Validation Tests
    // ========================================================================

    #[test]
    fn test_validate_openai_model_success() {
        let mapper = create_mapper();
        assert!(mapper.validate_model("gpt-4o", "openai").is_ok());
        assert!(mapper.validate_model("o3-mini", "openai").is_ok());
    }

    #[test]
    fn test_validate_gemini_model_success() {
        let mapper = create_mapper();
        assert!(mapper.validate_model("gemini-2.0-flash", "gemini").is_ok());
    }

    #[test]
    fn test_validate_ollama_model_success() {
        let mapper = create_mapper();
        assert!(mapper.validate_model("llama3", "ollama").is_ok());
    }

    #[test]
    fn test_validate_model_not_found() {
        let mapper = create_mapper();
        let result = mapper.validate_model("unknown-model", "openai");
        assert!(result.is_err());

        if let Err(ProxyError::ModelNotFound { model, provider, .. }) = result {
            assert_eq!(model, "unknown-model");
            assert_eq!(provider, "openai");
        } else {
            panic!("Expected ModelNotFound error");
        }
    }

    #[test]
    fn test_validate_wrong_provider() {
        let mapper = create_mapper();
        let result = mapper.validate_model("gpt-4o", "gemini");
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_unknown_provider() {
        let mapper = create_mapper();
        let result = mapper.validate_model("gpt-4o", "unknown");
        assert!(result.is_err());
    }

    // ========================================================================
    // End-to-End Resolution Tests
    // ========================================================================

    #[test]
    fn test_resolve_haiku_alias() {
        let mapper = create_mapper();
        let result = mapper.resolve_model("haiku", "openai").unwrap();
        assert_eq!(result, "openai/o3-mini");
    }

    #[test]
    fn test_resolve_sonnet_alias() {
        let mapper = create_mapper();
        let result = mapper.resolve_model("sonnet", "openai").unwrap();
        assert_eq!(result, "openai/o3-mini");
    }

    #[test]
    fn test_resolve_opus_alias() {
        let mapper = create_mapper();
        let result = mapper.resolve_model("opus", "openai").unwrap();
        assert_eq!(result, "openai/o3-mini");
    }

    #[test]
    fn test_resolve_claude_full_name() {
        let mapper = create_mapper();
        let result = mapper.resolve_model("claude-3-haiku-20240307", "openai").unwrap();
        assert_eq!(result, "openai/o3-mini");
    }

    #[test]
    fn test_resolve_with_prefix_stripped() {
        let mapper = create_mapper();
        let result = mapper.resolve_model("openai/gpt-4o", "gemini").unwrap();
        assert_eq!(result, "openai/gpt-4o");
    }

    #[test]
    fn test_resolve_without_prefix() {
        let mapper = create_mapper();
        let result = mapper.resolve_model("gpt-4o", "openai").unwrap();
        assert_eq!(result, "openai/gpt-4o");
    }

    #[test]
    fn test_resolve_auto_detect_provider() {
        let mapper = create_mapper();

        // OpenAI model auto-detected
        let result = mapper.resolve_model("gpt-4o-mini", "gemini").unwrap();
        assert_eq!(result, "openai/gpt-4o-mini");

        // Gemini model auto-detected
        let result = mapper.resolve_model("gemini-2.0-flash", "openai").unwrap();
        assert_eq!(result, "gemini/gemini-2.0-flash");

        // Ollama model auto-detected
        let result = mapper.resolve_model("llama3", "openai").unwrap();
        assert_eq!(result, "ollama/llama3");
    }

    #[test]
    fn test_resolve_unknown_model_error() {
        let mapper = create_mapper();
        let result = mapper.resolve_model("unknown-model", "openai");
        assert!(result.is_err());
    }

    #[test]
    fn test_resolve_case_insensitive_alias() {
        let mapper = create_mapper();
        let result1 = mapper.resolve_model("HAIKU", "openai").unwrap();
        let result2 = mapper.resolve_model("haiku", "openai").unwrap();
        assert_eq!(result1, result2);
    }

    // ========================================================================
    // All Models Tests
    // ========================================================================

    #[test]
    fn test_all_models() {
        let mapper = create_mapper();
        let all = mapper.all_models();

        assert!(all.contains(&"openai/gpt-4o".to_string()));
        assert!(all.contains(&"gemini/gemini-2.0-flash".to_string()));
        assert!(all.contains(&"ollama/llama3".to_string()));
        assert_eq!(all.len(), 4 + 2 + 4); // 4 openai + 2 gemini + 4 ollama
    }
}
