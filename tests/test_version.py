"""Tests for version compatibility checking."""

from unittest.mock import patch

import pytest


class TestVersionFunctions:
    """Tests for version checking utility functions."""

    def test_parse_version_valid(self):
        """Test parsing valid version strings."""
        from vllm_plugin_manager.core.version import parse_version

        version = parse_version("0.6.0")
        assert version is not None
        assert str(version) == "0.6.0"

    def test_parse_version_with_dev(self):
        """Test parsing version with dev suffix."""
        from vllm_plugin_manager.core.version import parse_version

        version = parse_version("0.1.1.dev123")
        assert version is not None

    def test_parse_version_invalid(self):
        """Test parsing invalid version string."""
        from vllm_plugin_manager.core.version import parse_version

        version = parse_version("not-a-version")
        assert version is None

    def test_check_version_constraint_satisfied(self):
        """Test version constraint that is satisfied."""
        from vllm_plugin_manager.core.version import check_version_constraint

        is_compatible, message = check_version_constraint("0.6.0", ">=0.5.0")
        assert is_compatible is True
        assert "satisfies" in message

    def test_check_version_constraint_not_satisfied(self):
        """Test version constraint that is not satisfied."""
        from vllm_plugin_manager.core.version import check_version_constraint

        is_compatible, message = check_version_constraint("0.4.0", ">=0.5.0")
        assert is_compatible is False
        assert "does not satisfy" in message

    def test_check_version_constraint_range(self):
        """Test version constraint with range."""
        from vllm_plugin_manager.core.version import check_version_constraint

        # Within range
        is_compatible, _ = check_version_constraint("0.6.0", ">=0.5.0,<1.0.0")
        assert is_compatible is True

        # Below range
        is_compatible, _ = check_version_constraint("0.4.0", ">=0.5.0,<1.0.0")
        assert is_compatible is False

        # Above range
        is_compatible, _ = check_version_constraint("1.0.0", ">=0.5.0,<1.0.0")
        assert is_compatible is False

    def test_check_version_constraint_exact(self):
        """Test exact version constraint."""
        from vllm_plugin_manager.core.version import check_version_constraint

        is_compatible, _ = check_version_constraint("0.6.0", "==0.6.0")
        assert is_compatible is True

        is_compatible, _ = check_version_constraint("0.6.1", "==0.6.0")
        assert is_compatible is False

    def test_check_version_constraint_invalid_constraint(self):
        """Test invalid version constraint."""
        from vllm_plugin_manager.core.version import check_version_constraint

        is_compatible, message = check_version_constraint("0.6.0", "invalid>>0.5")
        assert is_compatible is False
        assert "Invalid version constraint" in message

    def test_get_vllm_version_not_installed(self):
        """Test getting vLLM version when not installed."""
        from vllm_plugin_manager.core.version import get_vllm_version

        with patch("importlib.metadata.version") as mock_version:
            from importlib.metadata import PackageNotFoundError
            mock_version.side_effect = PackageNotFoundError("vllm")

            version = get_vllm_version()
            assert version is None

    def test_get_vllm_version_installed(self):
        """Test getting vLLM version when installed."""
        from vllm_plugin_manager.core.version import get_vllm_version

        with patch("importlib.metadata.version") as mock_version:
            mock_version.return_value = "0.6.0"

            version = get_vllm_version()
            assert version == "0.6.0"

    def test_check_vllm_compatibility_satisfied(self):
        """Test vLLM compatibility check when satisfied."""
        from vllm_plugin_manager.core.version import check_vllm_compatibility

        with patch("vllm_plugin_manager.core.version.get_vllm_version") as mock_get:
            mock_get.return_value = "0.6.0"

            is_compatible, _ = check_vllm_compatibility(">=0.5.0")
            assert is_compatible is True

    def test_check_vllm_compatibility_not_installed(self):
        """Test vLLM compatibility check when not installed."""
        from vllm_plugin_manager.core.version import check_vllm_compatibility

        with patch("vllm_plugin_manager.core.version.get_vllm_version") as mock_get:
            mock_get.return_value = None

            is_compatible, message = check_vllm_compatibility(">=0.5.0")
            assert is_compatible is False
            assert "not installed" in message


class TestVersionChecker:
    """Tests for VersionChecker class."""

    def test_version_checker_caches_version(self):
        """Test that VersionChecker caches the vLLM version."""
        from vllm_plugin_manager.core.version import VersionChecker

        with patch("vllm_plugin_manager.core.version.get_vllm_version") as mock_get:
            mock_get.return_value = "0.6.0"

            checker = VersionChecker()

            # Access version twice
            _ = checker.vllm_version
            _ = checker.vllm_version

            # Should only call get_vllm_version once
            assert mock_get.call_count == 1

    def test_version_checker_is_compatible_no_constraint(self):
        """Test is_compatible with no constraint."""
        from vllm_plugin_manager.core.version import VersionChecker

        checker = VersionChecker()
        is_compatible, message = checker.is_compatible(None)

        assert is_compatible is True
        assert "No version constraint" in message

    def test_version_checker_is_compatible_satisfied(self):
        """Test is_compatible when constraint is satisfied."""
        from vllm_plugin_manager.core.version import VersionChecker

        with patch("vllm_plugin_manager.core.version.get_vllm_version") as mock_get:
            mock_get.return_value = "0.6.0"

            checker = VersionChecker()
            is_compatible, _ = checker.is_compatible(">=0.5.0")

            assert is_compatible is True

    def test_version_checker_is_compatible_not_satisfied(self):
        """Test is_compatible when constraint is not satisfied."""
        from vllm_plugin_manager.core.version import VersionChecker

        with patch("vllm_plugin_manager.core.version.get_vllm_version") as mock_get:
            mock_get.return_value = "0.4.0"

            checker = VersionChecker()
            is_compatible, _ = checker.is_compatible(">=0.5.0")

            assert is_compatible is False

    def test_version_checker_check_plugin_compatibility(self):
        """Test check_plugin_compatibility method."""
        from vllm_plugin_manager.core.version import VersionChecker

        with patch("vllm_plugin_manager.core.version.get_vllm_version") as mock_get:
            mock_get.return_value = "0.6.0"

            checker = VersionChecker()

            # Compatible
            is_compatible, _ = checker.check_plugin_compatibility(
                "my-plugin", ">=0.5.0"
            )
            assert is_compatible is True

            # Not compatible
            is_compatible, _ = checker.check_plugin_compatibility(
                "my-plugin", ">=0.7.0"
            )
            assert is_compatible is False

            # No constraint
            is_compatible, _ = checker.check_plugin_compatibility(
                "my-plugin", None
            )
            assert is_compatible is True


class TestPluginManagerVersionCheck:
    """Tests for version checking in PluginManager."""

    def test_plugin_skipped_if_incompatible(self, temp_dir):
        """Test that incompatible plugins are skipped."""
        from vllm_plugin_manager.manager import PluginManager

        config_file = temp_dir / "plugins.yaml"
        config_file.write_text("""
plugins:
  - name: incompatible-plugin
    source: pypi
    package: incompatible-plugin
    enabled: true
    vllm_version: ">=99.0.0"
""")

        with patch("vllm_plugin_manager.core.version.get_vllm_version") as mock_get:
            mock_get.return_value = "0.6.0"

            manager = PluginManager(
                config_path=config_file,
                registry_dir=temp_dir / "registry",
            )

            results = manager.install_plugins()

            # Plugin should be skipped due to incompatibility
            assert "incompatible-plugin" in results
            is_success, message = results["incompatible-plugin"]
            assert is_success is False
            assert "Incompatible" in message

    def test_plugin_installed_if_compatible(self, temp_dir):
        """Test that compatible plugins are installed."""
        from vllm_plugin_manager.manager import PluginManager
        from unittest.mock import MagicMock

        config_file = temp_dir / "plugins.yaml"
        config_file.write_text("""
plugins:
  - name: compatible-plugin
    source: pypi
    package: compatible-plugin
    enabled: true
    vllm_version: ">=0.5.0"
""")

        with patch("vllm_plugin_manager.core.version.get_vllm_version") as mock_get:
            mock_get.return_value = "0.6.0"

            manager = PluginManager(
                config_path=config_file,
                registry_dir=temp_dir / "registry",
            )

            # Mock the installer to avoid actual installation
            manager.installer.install_from_spec = MagicMock(return_value=(True, "OK"))
            manager.installer.get_installed_version = MagicMock(return_value="1.0.0")

            results = manager.install_plugins()

            # Plugin should be installed
            assert "compatible-plugin" in results
            is_success, _ = results["compatible-plugin"]
            assert is_success is True

            # Installer should have been called
            manager.installer.install_from_spec.assert_called_once()

    def test_plugin_installed_if_no_version_constraint(self, temp_dir):
        """Test that plugins without version constraint are installed."""
        from vllm_plugin_manager.manager import PluginManager
        from unittest.mock import MagicMock

        config_file = temp_dir / "plugins.yaml"
        config_file.write_text("""
plugins:
  - name: no-constraint-plugin
    source: pypi
    package: no-constraint-plugin
    enabled: true
""")

        with patch("vllm_plugin_manager.core.version.get_vllm_version") as mock_get:
            mock_get.return_value = "0.6.0"

            manager = PluginManager(
                config_path=config_file,
                registry_dir=temp_dir / "registry",
            )

            # Mock the installer
            manager.installer.install_from_spec = MagicMock(return_value=(True, "OK"))
            manager.installer.get_installed_version = MagicMock(return_value="1.0.0")

            results = manager.install_plugins()

            # Plugin should be installed (no version constraint = always compatible)
            assert "no-constraint-plugin" in results
            is_success, _ = results["no-constraint-plugin"]
            assert is_success is True


class TestConfigVllmVersion:
    """Tests for vllm_version field in config."""

    def test_vllm_version_parsed_from_config(self, temp_dir):
        """Test that vllm_version is parsed from config."""
        from vllm_plugin_manager.config import PluginConfig

        config_file = temp_dir / "plugins.yaml"
        config_file.write_text("""
plugins:
  - name: test-plugin
    source: pypi
    package: test-plugin
    enabled: true
    vllm_version: ">=0.6.0,<1.0.0"
""")

        config = PluginConfig.from_file(config_file)
        plugin = config.plugins[0]

        assert plugin.vllm_version == ">=0.6.0,<1.0.0"

    def test_vllm_version_defaults_to_none(self, temp_dir):
        """Test that vllm_version defaults to None."""
        from vllm_plugin_manager.config import PluginConfig

        config_file = temp_dir / "plugins.yaml"
        config_file.write_text("""
plugins:
  - name: test-plugin
    source: pypi
    package: test-plugin
    enabled: true
""")

        config = PluginConfig.from_file(config_file)
        plugin = config.plugins[0]

        assert plugin.vllm_version is None
