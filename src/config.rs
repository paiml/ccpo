//! Configuration system for ccpo
//!
//! Implements hierarchical configuration loading with the following precedence (highest to lowest):
//! 1. CLI arguments
//! 2. Environment variables (CCPO_*)
//! 3. Local config file (./config.toml)
//! 4. User config file (~/.config/ccpo/config.toml)
//! 5. Embedded defaults
//!
//! # Example
//!
//! ```no_run
//! use ccpo::config::{AppConfig, Cli};
//! use clap::Parser;
//!
//! let cli = Cli::parse();
//! let config = AppConfig::load(&cli).expect("Failed to load configuration");
//! println!("Server will listen on {}:{}", config.server.host, config.server.port);
//! ```

use clap::Parser;
use config::{Config, ConfigError, Environment, File, FileFormat};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// CLI arguments
#[derive(Parser, Debug, Clone)]
#[command(author, version, about = "Claude Code Proxy for Other APIs", long_about = None)]
pub struct Cli {
    /// Config file path (overrides default locations)
    #[arg(short, long, env = "CCPO_CONFIG")]
    pub config: Option<PathBuf>,

    /// Server host to bind to
    #[arg(long, env = "CCPO_HOST")]
    pub host: Option<String>,

    /// Server port to bind to
    #[arg(short, long, env = "CCPO_PORT")]
    pub port: Option<u16>,

    /// Preferred provider (openai, google, ollama)
    #[arg(long, env = "CCPO_PREFERRED_PROVIDER")]
    pub preferred_provider: Option<String>,

    /// Big model (for Claude Sonnet)
    #[arg(long, env = "CCPO_BIG_MODEL")]
    pub big_model: Option<String>,

    /// Small model (for Claude Haiku)
    #[arg(long, env = "CCPO_SMALL_MODEL")]
    pub small_model: Option<String>,

    /// Log level (trace, debug, info, warn, error)
    #[arg(long, env = "CCPO_LOG_LEVEL")]
    pub log_level: Option<String>,

    /// OpenAI API key
    #[arg(long, env = "OPENAI_API_KEY")]
    pub openai_api_key: Option<String>,

    /// Gemini API key
    #[arg(long, env = "GEMINI_API_KEY")]
    pub gemini_api_key: Option<String>,

    /// Ollama API base URL
    #[arg(long, env = "OLLAMA_API_BASE")]
    pub ollama_api_base: Option<String>,
}

/// Complete application configuration
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AppConfig {
    /// Server configuration
    pub server: ServerConfig,

    /// Provider configuration
    pub providers: ProvidersConfig,

    /// Model configuration
    pub models: ModelsConfig,

    /// Logging configuration
    #[serde(default)]
    pub logging: LoggingConfig,
}

/// Server configuration
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ServerConfig {
    /// Host to bind to (e.g., "0.0.0.0", "127.0.0.1")
    pub host: String,

    /// Port to bind to (e.g., 8082)
    pub port: u16,

    /// Log level (trace, debug, info, warn, error)
    pub log_level: String,

    /// Log format (pretty or json)
    #[serde(default = "default_log_format")]
    pub log_format: String,
}

fn default_log_format() -> String {
    "pretty".to_string()
}

/// Provider configuration
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ProvidersConfig {
    /// Preferred provider (openai, google, ollama)
    pub preferred: String,

    /// OpenAI configuration
    pub openai: ProviderDetails,

    /// Google Gemini configuration
    pub google: ProviderDetails,

    /// Ollama configuration
    pub ollama: OllamaDetails,
}

/// Provider details (OpenAI, Gemini)
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ProviderDetails {
    /// API key
    pub api_key: String,

    /// Request timeout in seconds
    #[serde(default = "default_timeout")]
    pub timeout_secs: u64,

    /// Maximum retries
    #[serde(default = "default_max_retries")]
    pub max_retries: u32,
}

fn default_timeout() -> u64 {
    30
}

fn default_max_retries() -> u32 {
    3
}

/// Ollama-specific configuration
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OllamaDetails {
    /// API base URL
    pub api_base: String,

    /// Request timeout in seconds
    #[serde(default = "default_ollama_timeout")]
    pub timeout_secs: u64,

    /// Maximum retries
    #[serde(default = "default_max_retries")]
    pub max_retries: u32,
}

fn default_ollama_timeout() -> u64 {
    60
}

/// Model configuration
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ModelsConfig {
    /// Big model (maps Claude Sonnet)
    pub big: String,

    /// Small model (maps Claude Haiku)
    pub small: String,

    /// OpenAI models list
    pub openai_list: ModelList,

    /// Gemini models list
    pub gemini_list: ModelList,

    /// Ollama models list
    pub ollama_list: ModelList,
}

/// Model list
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ModelList {
    /// Supported models
    pub supported: Vec<String>,
}

/// Logging configuration
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct LoggingConfig {
    /// Phrases to filter from logs
    #[serde(default)]
    pub blocked_phrases: Vec<String>,
}

impl AppConfig {
    /// Load configuration from all sources with proper precedence
    ///
    /// Precedence (highest to lowest):
    /// 1. CLI arguments
    /// 2. Environment variables (CCPO_*)
    /// 3. Local config file (./config.toml)
    /// 4. User config file (~/.config/ccpo/config.toml)
    /// 5. Embedded defaults
    pub fn load(cli: &Cli) -> Result<Self, ConfigError> {
        let mut builder = Config::builder();

        // 1. Embedded defaults (lowest priority)
        let default_config = include_str!("../config/default.toml");
        builder = builder.add_source(File::from_str(default_config, FileFormat::Toml));

        // 2. User config file (~/.config/ccpo/config.toml)
        if let Some(user_config_path) = Self::default_user_config_path() {
            builder = builder.add_source(File::from(user_config_path).required(false));
        }

        // 3. Local config file (./config.toml)
        builder = builder.add_source(File::with_name("./config").required(false));

        // 4. Custom config file (if specified via CLI or env)
        if let Some(config_path) = &cli.config {
            builder = builder.add_source(File::from(config_path.as_path()));
        }

        // 5. Environment variables (CCPO_*)
        builder = builder.add_source(
            Environment::with_prefix("CCPO")
                .separator("__")
                .try_parsing(true),
        );

        // Build base config
        let mut config: AppConfig = builder.build()?.try_deserialize()?;

        // 6. Apply CLI overrides (highest priority)
        Self::apply_cli_overrides(&mut config, cli);

        // 7. Substitute environment variables in config values
        Self::substitute_env_vars(&mut config)?;

        // 8. Validate configuration
        config.validate()?;

        Ok(config)
    }

    /// Get default user config path (~/.config/ccpo/config.toml)
    fn default_user_config_path() -> Option<PathBuf> {
        dirs::config_dir().map(|dir| dir.join("ccpo").join("config.toml"))
    }

    /// Apply CLI argument overrides to configuration
    fn apply_cli_overrides(config: &mut AppConfig, cli: &Cli) {
        if let Some(ref host) = cli.host {
            config.server.host = host.clone();
        }
        if let Some(port) = cli.port {
            config.server.port = port;
        }
        if let Some(ref log_level) = cli.log_level {
            config.server.log_level = log_level.clone();
        }
        if let Some(ref provider) = cli.preferred_provider {
            config.providers.preferred = provider.clone();
        }
        if let Some(ref big_model) = cli.big_model {
            config.models.big = big_model.clone();
        }
        if let Some(ref small_model) = cli.small_model {
            config.models.small = small_model.clone();
        }
        if let Some(ref api_key) = cli.openai_api_key {
            config.providers.openai.api_key = api_key.clone();
        }
        if let Some(ref api_key) = cli.gemini_api_key {
            config.providers.google.api_key = api_key.clone();
        }
        if let Some(ref api_base) = cli.ollama_api_base {
            config.providers.ollama.api_base = api_base.clone();
        }
    }

    /// Substitute environment variables in configuration values
    ///
    /// Replaces ${VAR_NAME} patterns with environment variable values
    fn substitute_env_vars(&mut self) -> Result<(), ConfigError> {
        // Substitute in provider API keys
        self.providers.openai.api_key = Self::substitute_string(&self.providers.openai.api_key)?;
        self.providers.google.api_key = Self::substitute_string(&self.providers.google.api_key)?;

        Ok(())
    }

    /// Substitute environment variables in a single string
    fn substitute_string(value: &str) -> Result<String, ConfigError> {
        if !value.contains("${") {
            return Ok(value.to_string());
        }

        let mut result = value.to_string();
        let re = regex::Regex::new(r"\$\{([A-Z_][A-Z0-9_]*)\}").unwrap();

        for cap in re.captures_iter(value) {
            let full_match = &cap[0];
            let var_name = &cap[1];

            let var_value = std::env::var(var_name).map_err(|_| {
                ConfigError::Message(format!(
                    "Environment variable {} not found (required by config)",
                    var_name
                ))
            })?;

            result = result.replace(full_match, &var_value);
        }

        Ok(result)
    }

    /// Validate configuration
    fn validate(&self) -> Result<(), ConfigError> {
        // Validate provider
        let valid_providers = ["openai", "google", "ollama"];
        if !valid_providers.contains(&self.providers.preferred.as_str()) {
            return Err(ConfigError::Message(format!(
                "Invalid provider '{}'. Must be one of: {}",
                self.providers.preferred,
                valid_providers.join(", ")
            )));
        }

        // Validate log level
        let valid_levels = ["trace", "debug", "info", "warn", "error"];
        if !valid_levels.contains(&self.server.log_level.as_str()) {
            return Err(ConfigError::Message(format!(
                "Invalid log level '{}'. Must be one of: {}",
                self.server.log_level,
                valid_levels.join(", ")
            )));
        }

        // Validate port range
        if self.server.port == 0 {
            return Err(ConfigError::Message("Port must be greater than 0".to_string()));
        }

        // Validate API keys for selected provider
        match self.providers.preferred.as_str() {
            "openai" => {
                if self.providers.openai.api_key.is_empty() {
                    return Err(ConfigError::Message(
                        "OpenAI API key is required when using openai provider".to_string(),
                    ));
                }
            }
            "google" => {
                if self.providers.google.api_key.is_empty() {
                    return Err(ConfigError::Message(
                        "Gemini API key is required when using google provider".to_string(),
                    ));
                }
            }
            "ollama" => {
                if self.providers.ollama.api_base.is_empty() {
                    return Err(ConfigError::Message(
                        "Ollama API base URL is required when using ollama provider".to_string(),
                    ));
                }
            }
            _ => {}
        }

        Ok(())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;
    use std::env;
    use tempfile::NamedTempFile;
    use std::io::Write;

    /// Helper to create a temporary config file with .toml extension
    fn create_temp_config(content: &str) -> NamedTempFile {
        let mut file = tempfile::Builder::new()
            .suffix(".toml")
            .tempfile()
            .unwrap();
        file.write_all(content.as_bytes()).unwrap();
        file.flush().unwrap();
        file
    }

    /// Helper to set env var for test duration
    struct EnvGuard {
        key: String,
    }

    impl EnvGuard {
        fn new(key: &str, value: &str) -> Self {
            env::set_var(key, value);
            EnvGuard {
                key: key.to_string(),
            }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            env::remove_var(&self.key);
        }
    }

    #[test]
    #[serial]
    fn test_default_config_loads() {
        // Set required env vars for default config
        let _guard1 = EnvGuard::new("OPENAI_API_KEY", "test-openai-key");
        let _guard2 = EnvGuard::new("GEMINI_API_KEY", "test-gemini-key");

        let cli = Cli {
            config: None,
            host: None,
            port: None,
            preferred_provider: None,
            big_model: None,
            small_model: None,
            log_level: None,
            openai_api_key: None,
            gemini_api_key: None,
            ollama_api_base: None,
        };

        let config = AppConfig::load(&cli).unwrap();
        assert_eq!(config.server.host, "0.0.0.0");
        assert_eq!(config.server.port, 8082);
        assert_eq!(config.server.log_level, "info");
        assert_eq!(config.providers.preferred, "openai");
        assert_eq!(config.providers.openai.api_key, "test-openai-key");
        assert_eq!(config.providers.google.api_key, "test-gemini-key");
    }

    #[test]
    #[serial]
    fn test_cli_overrides_default() {
        // Set required env vars for default config
        let _guard1 = EnvGuard::new("OPENAI_API_KEY", "test-openai-key");
        let _guard2 = EnvGuard::new("GEMINI_API_KEY", "test-gemini-key");

        let cli = Cli {
            config: None,
            host: Some("127.0.0.1".to_string()),
            port: Some(9000),
            preferred_provider: Some("ollama".to_string()),
            big_model: Some("llama3:70b".to_string()),
            small_model: Some("llama3:8b".to_string()),
            log_level: Some("debug".to_string()),
            openai_api_key: None,
            gemini_api_key: None,
            ollama_api_base: Some("http://localhost:11434".to_string()),
        };

        let config = AppConfig::load(&cli).unwrap();
        assert_eq!(config.server.host, "127.0.0.1");
        assert_eq!(config.server.port, 9000);
        assert_eq!(config.server.log_level, "debug");
        assert_eq!(config.providers.preferred, "ollama");
        assert_eq!(config.models.big, "llama3:70b");
        assert_eq!(config.models.small, "llama3:8b");
    }

    #[test]
    fn test_custom_config_file() {
        let config_content = r#"
[server]
host = "192.168.1.1"
port = 7777
log_level = "warn"

[providers]
preferred = "google"

[providers.openai]
api_key = "test-openai-key"

[providers.google]
api_key = "test-gemini-key"

[providers.ollama]
api_base = "http://localhost:11434"

[models]
big = "gemini-2.5-pro-preview-03-25"
small = "gemini-2.0-flash"

[models.openai_list]
supported = ["gpt-4o"]

[models.gemini_list]
supported = ["gemini-2.0-flash"]

[models.ollama_list]
supported = ["llama3"]
"#;

        let temp_file = create_temp_config(config_content);
        let cli = Cli {
            config: Some(temp_file.path().to_path_buf()),
            host: None,
            port: None,
            preferred_provider: None,
            big_model: None,
            small_model: None,
            log_level: None,
            openai_api_key: None,
            gemini_api_key: None,
            ollama_api_base: None,
        };

        let config = AppConfig::load(&cli).unwrap();
        assert_eq!(config.server.host, "192.168.1.1");
        assert_eq!(config.server.port, 7777);
        assert_eq!(config.server.log_level, "warn");
        assert_eq!(config.providers.preferred, "google");
        assert_eq!(config.models.big, "gemini-2.5-pro-preview-03-25");
    }

    #[test]
    #[serial]
    fn test_env_var_substitution() {
        let _guard1 = EnvGuard::new("TEST_OPENAI_KEY", "env-openai-key");
        let _guard2 = EnvGuard::new("TEST_GEMINI_KEY", "env-gemini-key");

        let config_content = r#"
[server]
host = "0.0.0.0"
port = 8082
log_level = "info"

[providers]
preferred = "openai"

[providers.openai]
api_key = "${TEST_OPENAI_KEY}"

[providers.google]
api_key = "${TEST_GEMINI_KEY}"

[providers.ollama]
api_base = "http://localhost:11434"

[models]
big = "o3-mini"
small = "o3-mini"

[models.openai_list]
supported = ["o3-mini"]

[models.gemini_list]
supported = ["gemini-2.0-flash"]

[models.ollama_list]
supported = ["llama3"]
"#;

        let temp_file = create_temp_config(config_content);
        let cli = Cli {
            config: Some(temp_file.path().to_path_buf()),
            host: None,
            port: None,
            preferred_provider: None,
            big_model: None,
            small_model: None,
            log_level: None,
            openai_api_key: None,
            gemini_api_key: None,
            ollama_api_base: None,
        };

        let config = AppConfig::load(&cli).unwrap();
        assert_eq!(config.providers.openai.api_key, "env-openai-key");
        assert_eq!(config.providers.google.api_key, "env-gemini-key");
    }

    #[test]
    #[serial]
    fn test_env_var_substitution_missing_var() {
        // Make sure MISSING_VAR is not set
        env::remove_var("MISSING_VAR");

        let config_content = r#"
[server]
host = "0.0.0.0"
port = 8082
log_level = "info"

[providers]
preferred = "openai"

[providers.openai]
api_key = "${MISSING_VAR}"

[providers.google]
api_key = "test"

[providers.ollama]
api_base = "http://localhost:11434"

[models]
big = "o3-mini"
small = "o3-mini"

[models.openai_list]
supported = ["o3-mini"]

[models.gemini_list]
supported = ["gemini-2.0-flash"]

[models.ollama_list]
supported = ["llama3"]
"#;

        let temp_file = create_temp_config(config_content);
        let cli = Cli {
            config: Some(temp_file.path().to_path_buf()),
            host: None,
            port: None,
            preferred_provider: None,
            big_model: None,
            small_model: None,
            log_level: None,
            openai_api_key: None,
            gemini_api_key: None,
            ollama_api_base: None,
        };

        let result = AppConfig::load(&cli);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("MISSING_VAR not found"));
    }

    #[test]
    #[serial]
    fn test_validation_invalid_provider() {
        let _guard1 = EnvGuard::new("OPENAI_API_KEY", "test");
        let _guard2 = EnvGuard::new("GEMINI_API_KEY", "test");

        let cli = Cli {
            config: None,
            host: None,
            port: None,
            preferred_provider: Some("invalid".to_string()),
            big_model: None,
            small_model: None,
            log_level: None,
            openai_api_key: None,
            gemini_api_key: None,
            ollama_api_base: None,
        };

        let result = AppConfig::load(&cli);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("Invalid provider"), "Expected 'Invalid provider' error, got: {}", err_msg);
    }

    #[test]
    #[serial]
    fn test_validation_invalid_log_level() {
        let _guard1 = EnvGuard::new("OPENAI_API_KEY", "test");
        let _guard2 = EnvGuard::new("GEMINI_API_KEY", "test");

        let cli = Cli {
            config: None,
            host: None,
            port: None,
            preferred_provider: None,
            big_model: None,
            small_model: None,
            log_level: Some("invalid".to_string()),
            openai_api_key: None,
            gemini_api_key: None,
            ollama_api_base: None,
        };

        let result = AppConfig::load(&cli);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("Invalid log level"), "Expected 'Invalid log level' error, got: {}", err_msg);
    }

    #[test]
    fn test_validation_missing_openai_key() {
        let config_content = r#"
[server]
host = "0.0.0.0"
port = 8082
log_level = "info"

[providers]
preferred = "openai"

[providers.openai]
api_key = ""

[providers.google]
api_key = "test"

[providers.ollama]
api_base = "http://localhost:11434"

[models]
big = "o3-mini"
small = "o3-mini"

[models.openai_list]
supported = ["o3-mini"]

[models.gemini_list]
supported = ["gemini-2.0-flash"]

[models.ollama_list]
supported = ["llama3"]
"#;

        let temp_file = create_temp_config(config_content);
        let cli = Cli {
            config: Some(temp_file.path().to_path_buf()),
            host: None,
            port: None,
            preferred_provider: None,
            big_model: None,
            small_model: None,
            log_level: None,
            openai_api_key: None,
            gemini_api_key: None,
            ollama_api_base: None,
        };

        let result = AppConfig::load(&cli);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("OpenAI API key"));
    }

    #[test]
    #[serial]
    fn test_validation_zero_port() {
        let _guard1 = EnvGuard::new("OPENAI_API_KEY", "test");
        let _guard2 = EnvGuard::new("GEMINI_API_KEY", "test");

        let cli = Cli {
            config: None,
            host: None,
            port: Some(0),
            preferred_provider: None,
            big_model: None,
            small_model: None,
            log_level: None,
            openai_api_key: None,
            gemini_api_key: None,
            ollama_api_base: None,
        };

        let result = AppConfig::load(&cli);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("Port must be greater than 0"), "Expected 'Port must be greater than 0' error, got: {}", err_msg);
    }

    #[test]
    fn test_cli_precedence_over_config_file() {
        let config_content = r#"
[server]
host = "192.168.1.1"
port = 7777
log_level = "warn"

[providers]
preferred = "google"

[providers.openai]
api_key = "file-key"

[providers.google]
api_key = "file-key"

[providers.ollama]
api_base = "http://localhost:11434"

[models]
big = "file-model"
small = "file-model"

[models.openai_list]
supported = ["gpt-4o"]

[models.gemini_list]
supported = ["gemini-2.0-flash"]

[models.ollama_list]
supported = ["llama3"]
"#;

        let temp_file = create_temp_config(config_content);
        let cli = Cli {
            config: Some(temp_file.path().to_path_buf()),
            host: Some("127.0.0.1".to_string()),
            port: Some(9000),
            preferred_provider: Some("openai".to_string()),
            big_model: Some("cli-model".to_string()),
            small_model: Some("cli-model".to_string()),
            log_level: Some("debug".to_string()),
            openai_api_key: Some("cli-key".to_string()),
            gemini_api_key: None,
            ollama_api_base: None,
        };

        let config = AppConfig::load(&cli).unwrap();
        // CLI should override file
        assert_eq!(config.server.host, "127.0.0.1");
        assert_eq!(config.server.port, 9000);
        assert_eq!(config.server.log_level, "debug");
        assert_eq!(config.providers.preferred, "openai");
        assert_eq!(config.models.big, "cli-model");
        assert_eq!(config.models.small, "cli-model");
        assert_eq!(config.providers.openai.api_key, "cli-key");
    }

    #[test]
    #[serial]
    fn test_default_values() {
        let _guard1 = EnvGuard::new("OPENAI_API_KEY", "test");
        let _guard2 = EnvGuard::new("GEMINI_API_KEY", "test");

        let cli = Cli {
            config: None,
            host: None,
            port: None,
            preferred_provider: None,
            big_model: None,
            small_model: None,
            log_level: None,
            openai_api_key: None,
            gemini_api_key: None,
            ollama_api_base: None,
        };

        let config = AppConfig::load(&cli).unwrap();
        assert_eq!(config.server.log_format, "pretty");
        assert_eq!(config.providers.openai.timeout_secs, 30);
        assert_eq!(config.providers.openai.max_retries, 3);
        assert_eq!(config.providers.ollama.timeout_secs, 60);
    }

    #[test]
    #[serial]
    fn test_model_lists_loaded() {
        let _guard1 = EnvGuard::new("OPENAI_API_KEY", "test");
        let _guard2 = EnvGuard::new("GEMINI_API_KEY", "test");

        let cli = Cli {
            config: None,
            host: None,
            port: None,
            preferred_provider: None,
            big_model: None,
            small_model: None,
            log_level: None,
            openai_api_key: None,
            gemini_api_key: None,
            ollama_api_base: None,
        };

        let config = AppConfig::load(&cli).unwrap();
        assert!(!config.models.openai_list.supported.is_empty());
        assert!(config
            .models
            .openai_list
            .supported
            .contains(&"o3-mini".to_string()));
        assert!(!config.models.gemini_list.supported.is_empty());
        assert!(!config.models.ollama_list.supported.is_empty());
    }

    #[test]
    fn test_substitute_string_no_vars() {
        let result = AppConfig::substitute_string("plain-text").unwrap();
        assert_eq!(result, "plain-text");
    }

    #[test]
    #[serial]
    fn test_substitute_string_with_var() {
        let _guard = EnvGuard::new("TEST_VAR", "test-value");
        let result = AppConfig::substitute_string("prefix-${TEST_VAR}-suffix").unwrap();
        assert_eq!(result, "prefix-test-value-suffix");
    }

    #[test]
    #[serial]
    fn test_multiple_env_vars_in_string() {
        let _guard1 = EnvGuard::new("VAR1", "value1");
        let _guard2 = EnvGuard::new("VAR2", "value2");
        let result = AppConfig::substitute_string("${VAR1}-${VAR2}").unwrap();
        assert_eq!(result, "value1-value2");
    }
}
