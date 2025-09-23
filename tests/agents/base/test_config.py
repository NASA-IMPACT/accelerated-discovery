"""Test cases for BaseAgentConfig."""

from pydantic import AnyUrl

from akd.agents._base import BaseAgentConfig
from akd.configs.project import CONFIG

from .conftest import AgentTestCustomConfig, create_config_with_overrides


class TestBaseAgentConfigMethods:
    """Test BaseAgentConfig functionality."""

    def test_default_config_initialization(self):
        """Test default configuration initialization."""
        config = BaseAgentConfig()

        # Test default values from CONFIG
        assert config.base_url == CONFIG.model_config_settings.base_url
        assert config.api_key == CONFIG.model_config_settings.api_keys.openai
        assert config.model_name == CONFIG.model_config_settings.model_name
        assert config.temperature == 0.0
        assert config.system_prompt is not None

    def test_custom_config_initialization(self):
        """Test custom configuration initialization."""
        custom_config = AgentTestCustomConfig(
            model_name="gpt-4o-mini",  # Use high-capacity model for testing large max_tokens
            temperature=0.7,
            custom_field="test_value",
            custom_temperature=0.8,
        )

        assert custom_config.model_name == "gpt-4o-mini"
        assert custom_config.temperature == 0.7
        assert custom_config.custom_field == "test_value"
        assert custom_config.custom_temperature == 0.8

    def test_config_field_validation(self):
        """Test configuration field validation."""
        # Test with invalid base_url type should still work (AnyUrl handles conversion)
        config = BaseAgentConfig(
            base_url="https://custom.api.com/v1",
            temperature=0.5,
        )
        assert isinstance(config.base_url, AnyUrl)
        assert config.temperature == 0.5

    def test_token_management_fields(self):
        """Test token management configuration fields."""
        config = BaseAgentConfig(
            model_name="gpt-4o-mini",
            max_tokens=25000,
            trim_ratio=0.6,
            enable_trimming=False,
        )

        assert config.max_tokens == 25000
        assert config.trim_ratio == 0.6
        assert config.enable_trimming is False

    def test_stateless_field(self):
        """Test stateless configuration field."""
        # Test default (stateless=True)
        config = BaseAgentConfig()
        assert config.stateless is True

        # Test explicit stateless=False
        config = BaseAgentConfig(stateless=False)
        assert config.stateless is False

    def test_config_override_helper(self, default_config):
        """Test the config override helper function."""
        overridden_config = create_config_with_overrides(
            default_config,
            temperature=0.9,
            enable_trimming=False,
        )

        assert overridden_config.temperature == 0.9
        assert overridden_config.enable_trimming is False
        # Other fields should remain the same
        assert overridden_config.model_name == default_config.model_name
        assert overridden_config.max_tokens == default_config.max_tokens

    def test_none_config_handling(self):
        """Test handling of None values in config."""
        config = BaseAgentConfig(
            api_key=None,
            base_url=None,
            model_name=None,
        )

        # When explicitly set to None, they should remain None (not defaults)
        assert config.api_key is None
        assert config.base_url is None
        assert config.model_name is None

    def test_config_serialization(self):
        """Test config serialization and deserialization."""
        original_config = BaseAgentConfig(
            model_name="gpt-4o-mini",
            temperature=0.5,
            max_tokens=30000,
            trim_ratio=0.8,
            enable_trimming=True,
            stateless=False,
        )

        # Test model_dump
        config_dict = original_config.model_dump()
        assert isinstance(config_dict, dict)
        assert config_dict["model_name"] == "gpt-4o-mini"
        assert config_dict["temperature"] == 0.5

        # Test reconstruction from dict
        reconstructed_config = BaseAgentConfig(**config_dict)
        assert reconstructed_config.model_name == original_config.model_name
        assert reconstructed_config.temperature == original_config.temperature
        assert reconstructed_config.max_tokens == original_config.max_tokens
        assert reconstructed_config.trim_ratio == original_config.trim_ratio
        assert reconstructed_config.enable_trimming == original_config.enable_trimming
        assert reconstructed_config.stateless == original_config.stateless
