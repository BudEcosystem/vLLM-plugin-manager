"""Tests for configuration loading and parsing."""

import os
from pathlib import Path

import pytest


class TestPluginConfig:
    """Tests for PluginConfig class."""

    def test_load_from_yaml_file(self, sample_plugins_yaml: Path):
        """Test loading plugins from a YAML config file."""
        from vllm_plugin_manager.config import PluginConfig

        config = PluginConfig.from_file(sample_plugins_yaml)

        assert len(config.plugins) == 3
        assert config.plugins[0].name == "vllm-entropy-decoder"
        assert config.plugins[0].source == "pypi"
        assert config.plugins[0].package == "vllm-entropy-decoder"

    def test_load_filters_disabled_plugins(self, sample_plugins_yaml: Path):
        """Test that disabled plugins are filtered out when requested."""
        from vllm_plugin_manager.config import PluginConfig

        config = PluginConfig.from_file(sample_plugins_yaml)
        enabled_plugins = config.get_enabled_plugins()

        assert len(enabled_plugins) == 2
        assert all(p.enabled for p in enabled_plugins)

    def test_pypi_plugin_spec(self, sample_plugins_yaml: Path):
        """Test PyPI plugin specification parsing."""
        from vllm_plugin_manager.config import PluginConfig

        config = PluginConfig.from_file(sample_plugins_yaml)
        pypi_plugin = config.plugins[0]

        assert pypi_plugin.source == "pypi"
        assert pypi_plugin.package == "vllm-entropy-decoder"
        assert pypi_plugin.version == ">=0.1.0"

    def test_git_plugin_spec(self, sample_plugins_yaml: Path):
        """Test Git plugin specification parsing."""
        from vllm_plugin_manager.config import PluginConfig

        config = PluginConfig.from_file(sample_plugins_yaml)
        git_plugin = config.plugins[1]

        assert git_plugin.source == "git"
        assert git_plugin.url == "https://github.com/user/my-plugin.git"
        assert git_plugin.ref == "main"

    def test_load_from_env_var(self, temp_dir: Path, monkeypatch: pytest.MonkeyPatch):
        """Test loading config path from environment variable."""
        from vllm_plugin_manager.config import PluginConfig, get_config_path

        config_file = temp_dir / "custom-plugins.yaml"
        config_file.write_text("""
plugins:
  - name: test-plugin
    source: pypi
    package: test-plugin
    enabled: true
""")
        monkeypatch.setenv("VLLM_PLUGIN_CONFIG", str(config_file))

        config_path = get_config_path()
        assert config_path == config_file

        config = PluginConfig.from_file(config_path)
        assert len(config.plugins) == 1
        assert config.plugins[0].name == "test-plugin"

    def test_default_config_path(self, monkeypatch: pytest.MonkeyPatch, temp_dir: Path):
        """Test default config path resolution."""
        from vllm_plugin_manager.config import get_config_path

        monkeypatch.delenv("VLLM_PLUGIN_CONFIG", raising=False)
        monkeypatch.setenv("HOME", str(temp_dir))

        # Create the default config location
        config_dir = temp_dir / ".config" / "vllm"
        config_dir.mkdir(parents=True)
        config_file = config_dir / "plugins.yaml"
        config_file.write_text("plugins: []")

        config_path = get_config_path()
        assert config_path == config_file

    def test_missing_config_file_returns_none(self, monkeypatch: pytest.MonkeyPatch, temp_dir: Path):
        """Test that missing config file returns None."""
        from vllm_plugin_manager.config import get_config_path

        monkeypatch.delenv("VLLM_PLUGIN_CONFIG", raising=False)
        monkeypatch.setenv("HOME", str(temp_dir))

        config_path = get_config_path()
        assert config_path is None

    def test_empty_config_file(self, temp_dir: Path):
        """Test handling of empty config file."""
        from vllm_plugin_manager.config import PluginConfig

        config_file = temp_dir / "empty.yaml"
        config_file.write_text("plugins: []")

        config = PluginConfig.from_file(config_file)
        assert len(config.plugins) == 0

    def test_invalid_yaml_raises_error(self, temp_dir: Path):
        """Test that invalid YAML raises an error."""
        from vllm_plugin_manager.config import PluginConfig, ConfigError

        config_file = temp_dir / "invalid.yaml"
        config_file.write_text("plugins: [invalid yaml content")

        with pytest.raises(ConfigError):
            PluginConfig.from_file(config_file)

    def test_missing_required_fields_raises_error(self, temp_dir: Path):
        """Test that missing required fields raise an error."""
        from vllm_plugin_manager.config import PluginConfig, ConfigError

        config_file = temp_dir / "missing-fields.yaml"
        config_file.write_text("""
plugins:
  - name: incomplete-plugin
    # missing source field
""")

        with pytest.raises(ConfigError):
            PluginConfig.from_file(config_file)


class TestPluginSpec:
    """Tests for PluginSpec dataclass."""

    def test_pypi_install_spec(self, sample_plugins_yaml: Path):
        """Test generating pip install spec for PyPI plugins."""
        from vllm_plugin_manager.config import PluginConfig

        config = PluginConfig.from_file(sample_plugins_yaml)
        plugin = config.plugins[0]

        install_spec = plugin.get_install_spec()
        assert install_spec == "vllm-entropy-decoder>=0.1.0"

    def test_pypi_install_spec_no_version(self, temp_dir: Path):
        """Test pip install spec without version constraint."""
        from vllm_plugin_manager.config import PluginConfig

        config_file = temp_dir / "plugins.yaml"
        config_file.write_text("""
plugins:
  - name: simple-plugin
    source: pypi
    package: simple-plugin
    enabled: true
""")

        config = PluginConfig.from_file(config_file)
        plugin = config.plugins[0]

        install_spec = plugin.get_install_spec()
        assert install_spec == "simple-plugin"

    def test_git_install_spec(self, sample_plugins_yaml: Path):
        """Test generating pip install spec for Git plugins."""
        from vllm_plugin_manager.config import PluginConfig

        config = PluginConfig.from_file(sample_plugins_yaml)
        plugin = config.plugins[1]

        install_spec = plugin.get_install_spec()
        assert "git+https://github.com/user/my-plugin.git@main" in install_spec

    def test_git_install_spec_with_subdirectory(self, temp_dir: Path):
        """Test Git install spec with subdirectory."""
        from vllm_plugin_manager.config import PluginConfig

        config_file = temp_dir / "plugins.yaml"
        config_file.write_text("""
plugins:
  - name: monorepo-plugin
    source: git
    url: https://github.com/user/monorepo.git
    ref: v1.0.0
    subdirectory: plugins/my-plugin
    enabled: true
""")

        config = PluginConfig.from_file(config_file)
        plugin = config.plugins[0]

        install_spec = plugin.get_install_spec()
        assert "git+https://github.com/user/monorepo.git@v1.0.0" in install_spec
        assert "#subdirectory=plugins/my-plugin" in install_spec

    def test_local_install_spec(self, temp_dir: Path):
        """Test generating pip install spec for local plugins."""
        from vllm_plugin_manager.config import PluginConfig

        plugin_dir = temp_dir / "my-local-plugin"
        plugin_dir.mkdir()

        config_file = temp_dir / "plugins.yaml"
        config_file.write_text(f"""
plugins:
  - name: local-plugin
    source: local
    path: {plugin_dir}
    editable: true
    enabled: true
""")

        config = PluginConfig.from_file(config_file)
        plugin = config.plugins[0]

        install_spec = plugin.get_install_spec()
        assert str(plugin_dir) in install_spec

    def test_plugin_id_generation(self, sample_plugins_yaml: Path):
        """Test that plugin IDs are generated correctly."""
        from vllm_plugin_manager.config import PluginConfig

        config = PluginConfig.from_file(sample_plugins_yaml)

        # PyPI plugin ID is the package name
        assert config.plugins[0].plugin_id == "vllm-entropy-decoder"

        # Git plugin ID is the name
        assert config.plugins[1].plugin_id == "my-custom-plugin"
