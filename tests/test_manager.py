"""Tests for Plugin Manager."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestPluginManagerEnvVars:
    """Tests for environment variable handling in PluginManager."""

    def test_apply_env_vars_sets_variables(self, temp_dir: Path, monkeypatch: pytest.MonkeyPatch):
        """Test that env vars from plugin config are set in os.environ."""
        from vllm_plugin_manager.config import PluginSpec
        from vllm_plugin_manager.manager import PluginManager

        # Create a config file with env vars
        config_file = temp_dir / "plugins.yaml"
        config_file.write_text("""
plugins:
  - name: test-plugin
    source: pypi
    package: test-plugin
    enabled: true
    env:
      TEST_VAR_ONE: "value1"
      TEST_VAR_TWO: "value2"
""")

        # Clean up any existing env vars
        monkeypatch.delenv("TEST_VAR_ONE", raising=False)
        monkeypatch.delenv("TEST_VAR_TWO", raising=False)

        manager = PluginManager(
            config_path=config_file,
            registry_dir=temp_dir / "registry",
        )

        # Get the plugin spec
        spec = manager.config.plugins[0]

        # Apply env vars
        manager._apply_env_vars(spec)

        # Verify env vars are set
        assert os.environ.get("TEST_VAR_ONE") == "value1"
        assert os.environ.get("TEST_VAR_TWO") == "value2"

    def test_apply_env_vars_no_env_field(self, temp_dir: Path):
        """Test that _apply_env_vars handles plugins without env field."""
        from vllm_plugin_manager.config import PluginSpec
        from vllm_plugin_manager.manager import PluginManager

        # Create a config file without env vars
        config_file = temp_dir / "plugins.yaml"
        config_file.write_text("""
plugins:
  - name: test-plugin
    source: pypi
    package: test-plugin
    enabled: true
""")

        manager = PluginManager(
            config_path=config_file,
            registry_dir=temp_dir / "registry",
        )

        # Get the plugin spec
        spec = manager.config.plugins[0]

        # Should not raise an error
        manager._apply_env_vars(spec)

    def test_install_plugins_applies_env_vars_first(self, temp_dir: Path, monkeypatch: pytest.MonkeyPatch):
        """Test that env vars are applied before plugin installation."""
        from vllm_plugin_manager.manager import PluginManager

        # Create a config file with env vars
        config_file = temp_dir / "plugins.yaml"
        config_file.write_text("""
plugins:
  - name: test-plugin
    source: pypi
    package: test-plugin
    enabled: true
    env:
      INSTALL_TEST_VAR: "before_install"
""")

        monkeypatch.delenv("INSTALL_TEST_VAR", raising=False)

        manager = PluginManager(
            config_path=config_file,
            registry_dir=temp_dir / "registry",
        )

        # Track when env var was set relative to installation
        env_var_set_before_install = []

        original_install = manager.installer.install_from_spec

        def mock_install(spec):
            # Check if env var is set at time of installation
            env_var_set_before_install.append(os.environ.get("INSTALL_TEST_VAR"))
            return True, "Mocked install"

        manager.installer.install_from_spec = mock_install

        # Run install
        manager.install_plugins()

        # Verify env var was set before install was called
        assert len(env_var_set_before_install) == 1
        assert env_var_set_before_install[0] == "before_install"

    def test_env_vars_with_integer_values(self, temp_dir: Path, monkeypatch: pytest.MonkeyPatch):
        """Test that integer env values are converted to strings."""
        from vllm_plugin_manager.manager import PluginManager

        # Create a config file with integer env value (YAML may parse as int)
        config_file = temp_dir / "plugins.yaml"
        config_file.write_text("""
plugins:
  - name: test-plugin
    source: pypi
    package: test-plugin
    enabled: true
    env:
      NUMERIC_VAR: 123
""")

        monkeypatch.delenv("NUMERIC_VAR", raising=False)

        manager = PluginManager(
            config_path=config_file,
            registry_dir=temp_dir / "registry",
        )

        spec = manager.config.plugins[0]
        manager._apply_env_vars(spec)

        # Should be converted to string
        assert os.environ.get("NUMERIC_VAR") == "123"

    def test_multiple_plugins_env_vars(self, temp_dir: Path, monkeypatch: pytest.MonkeyPatch):
        """Test that env vars from multiple plugins are all applied."""
        from vllm_plugin_manager.manager import PluginManager

        config_file = temp_dir / "plugins.yaml"
        config_file.write_text("""
plugins:
  - name: plugin-one
    source: pypi
    package: plugin-one
    enabled: true
    env:
      PLUGIN_ONE_VAR: "one"
  - name: plugin-two
    source: pypi
    package: plugin-two
    enabled: true
    env:
      PLUGIN_TWO_VAR: "two"
""")

        monkeypatch.delenv("PLUGIN_ONE_VAR", raising=False)
        monkeypatch.delenv("PLUGIN_TWO_VAR", raising=False)

        manager = PluginManager(
            config_path=config_file,
            registry_dir=temp_dir / "registry",
        )

        # Mock the installer to avoid actual installation
        manager.installer.install_from_spec = MagicMock(return_value=(True, "Mocked"))
        manager.registry.is_installed = MagicMock(return_value=False)

        manager.install_plugins()

        # Both env vars should be set
        assert os.environ.get("PLUGIN_ONE_VAR") == "one"
        assert os.environ.get("PLUGIN_TWO_VAR") == "two"
