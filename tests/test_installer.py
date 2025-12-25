"""Tests for package installer (PyPI/Git installation)."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestPackageInstaller:
    """Tests for PackageInstaller class."""

    def test_install_pypi_package(self, temp_dir: Path):
        """Test installing a package from PyPI."""
        from vllm_plugin_manager.sources.installer import PackageInstaller

        installer = PackageInstaller()

        with patch.object(installer, "_run_pip") as mock_pip:
            mock_pip.return_value = (True, "Successfully installed test-package-1.0.0")

            success, message = installer.install_pypi("test-package", version=">=1.0.0")

            assert success is True
            mock_pip.assert_called_once()
            call_args = mock_pip.call_args[0][0]
            assert "install" in call_args
            assert "test-package>=1.0.0" in call_args

    def test_install_pypi_package_no_version(self, temp_dir: Path):
        """Test installing a package from PyPI without version constraint."""
        from vllm_plugin_manager.sources.installer import PackageInstaller

        installer = PackageInstaller()

        with patch.object(installer, "_run_pip") as mock_pip:
            mock_pip.return_value = (True, "Successfully installed test-package-1.0.0")

            success, message = installer.install_pypi("test-package")

            assert success is True
            call_args = mock_pip.call_args[0][0]
            assert "test-package" in call_args
            # Should not have version specifier
            assert not any(">=" in arg or "==" in arg for arg in call_args if "test-package" in arg)

    def test_install_pypi_package_failure(self):
        """Test handling PyPI installation failure."""
        from vllm_plugin_manager.sources.installer import PackageInstaller

        installer = PackageInstaller()

        with patch.object(installer, "_run_pip") as mock_pip:
            mock_pip.return_value = (False, "ERROR: No matching distribution found")

            success, message = installer.install_pypi("nonexistent-package-xyz")

            assert success is False
            assert "ERROR" in message or "error" in message.lower()

    def test_install_git_package(self):
        """Test installing a package from Git repository."""
        from vllm_plugin_manager.sources.installer import PackageInstaller

        installer = PackageInstaller()

        with patch.object(installer, "_run_pip") as mock_pip:
            mock_pip.return_value = (True, "Successfully installed my-plugin")

            success, message = installer.install_git(
                url="https://github.com/user/my-plugin.git",
                ref="main",
            )

            assert success is True
            call_args = mock_pip.call_args[0][0]
            assert "install" in call_args
            assert any("git+https://github.com/user/my-plugin.git@main" in arg for arg in call_args)

    def test_install_git_package_with_subdirectory(self):
        """Test installing a package from Git repo subdirectory."""
        from vllm_plugin_manager.sources.installer import PackageInstaller

        installer = PackageInstaller()

        with patch.object(installer, "_run_pip") as mock_pip:
            mock_pip.return_value = (True, "Successfully installed my-plugin")

            success, message = installer.install_git(
                url="https://github.com/user/monorepo.git",
                ref="v1.0.0",
                subdirectory="plugins/my-plugin",
            )

            assert success is True
            call_args = mock_pip.call_args[0][0]
            # Check that subdirectory is included
            assert any("#subdirectory=plugins/my-plugin" in arg for arg in call_args)

    def test_install_git_package_no_ref(self):
        """Test installing from Git without specific ref (uses default branch)."""
        from vllm_plugin_manager.sources.installer import PackageInstaller

        installer = PackageInstaller()

        with patch.object(installer, "_run_pip") as mock_pip:
            mock_pip.return_value = (True, "Successfully installed my-plugin")

            success, message = installer.install_git(
                url="https://github.com/user/my-plugin.git",
            )

            assert success is True
            call_args = mock_pip.call_args[0][0]
            # Should not have @ref in the URL
            assert any("git+https://github.com/user/my-plugin.git" in arg for arg in call_args)

    def test_install_local_package(self, temp_dir: Path):
        """Test installing a local package."""
        from vllm_plugin_manager.sources.installer import PackageInstaller

        # Create a minimal package structure
        plugin_dir = temp_dir / "my-local-plugin"
        plugin_dir.mkdir()
        (plugin_dir / "pyproject.toml").write_text("""
[project]
name = "my-local-plugin"
version = "0.1.0"
""")
        (plugin_dir / "my_local_plugin.py").write_text("# plugin code")

        installer = PackageInstaller()

        with patch.object(installer, "_run_pip") as mock_pip:
            mock_pip.return_value = (True, "Successfully installed my-local-plugin")

            success, message = installer.install_local(path=plugin_dir, editable=True)

            assert success is True
            call_args = mock_pip.call_args[0][0]
            assert "install" in call_args
            assert "-e" in call_args  # editable flag
            assert str(plugin_dir) in call_args

    def test_install_local_package_non_editable(self, temp_dir: Path):
        """Test installing a local package in non-editable mode."""
        from vllm_plugin_manager.sources.installer import PackageInstaller

        plugin_dir = temp_dir / "my-local-plugin"
        plugin_dir.mkdir()

        installer = PackageInstaller()

        with patch.object(installer, "_run_pip") as mock_pip:
            mock_pip.return_value = (True, "Successfully installed my-local-plugin")

            success, message = installer.install_local(path=plugin_dir, editable=False)

            assert success is True
            call_args = mock_pip.call_args[0][0]
            assert "-e" not in call_args

    def test_uninstall_package(self):
        """Test uninstalling a package."""
        from vllm_plugin_manager.sources.installer import PackageInstaller

        installer = PackageInstaller()

        with patch.object(installer, "_run_pip") as mock_pip:
            mock_pip.return_value = (True, "Successfully uninstalled test-package")

            success, message = installer.uninstall("test-package")

            assert success is True
            call_args = mock_pip.call_args[0][0]
            assert "uninstall" in call_args
            assert "-y" in call_args  # auto-confirm
            assert "test-package" in call_args

    def test_is_package_installed(self):
        """Test checking if a package is installed."""
        from vllm_plugin_manager.sources.installer import PackageInstaller

        installer = PackageInstaller()

        # pytest should be installed (we're running tests with it)
        assert installer.is_installed("pytest") is True

        # Random nonexistent package
        assert installer.is_installed("nonexistent-package-xyz-123") is False

    def test_get_installed_version(self):
        """Test getting installed package version."""
        from vllm_plugin_manager.sources.installer import PackageInstaller

        installer = PackageInstaller()

        # pytest should be installed
        version = installer.get_installed_version("pytest")
        assert version is not None
        assert len(version) > 0

        # Nonexistent package
        version = installer.get_installed_version("nonexistent-package-xyz-123")
        assert version is None


class TestPipRunner:
    """Tests for pip command execution."""

    def test_run_pip_success(self):
        """Test successful pip command execution."""
        from vllm_plugin_manager.sources.installer import PackageInstaller

        installer = PackageInstaller()

        # Run a simple pip command that should succeed
        success, output = installer._run_pip(["--version"])

        assert success is True
        assert "pip" in output.lower()

    def test_run_pip_failure(self):
        """Test pip command failure handling."""
        from vllm_plugin_manager.sources.installer import PackageInstaller

        installer = PackageInstaller()

        # Run a pip command that should fail
        success, output = installer._run_pip(["install", "nonexistent-package-xyz-abc-123"])

        assert success is False

    def test_run_pip_timeout(self):
        """Test pip command timeout handling."""
        from vllm_plugin_manager.sources.installer import PackageInstaller

        installer = PackageInstaller(timeout=0.001)  # Very short timeout

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="pip", timeout=0.001)

            success, output = installer._run_pip(["install", "some-package"])

            assert success is False
            assert "timed out" in output.lower()


class TestInstallFromSpec:
    """Tests for installing from PluginSpec."""

    def test_install_from_pypi_spec(self, temp_dir: Path):
        """Test installing from a PyPI PluginSpec."""
        from vllm_plugin_manager.config import PluginConfig
        from vllm_plugin_manager.sources.installer import PackageInstaller

        config_file = temp_dir / "plugins.yaml"
        config_file.write_text("""
plugins:
  - name: test-plugin
    source: pypi
    package: test-plugin
    version: ">=1.0.0"
    enabled: true
""")

        config = PluginConfig.from_file(config_file)
        plugin_spec = config.plugins[0]

        installer = PackageInstaller()

        with patch.object(installer, "_run_pip") as mock_pip:
            mock_pip.return_value = (True, "Successfully installed")

            success, message = installer.install_from_spec(plugin_spec)

            assert success is True

    def test_install_from_git_spec(self, temp_dir: Path):
        """Test installing from a Git PluginSpec."""
        from vllm_plugin_manager.config import PluginConfig
        from vllm_plugin_manager.sources.installer import PackageInstaller

        config_file = temp_dir / "plugins.yaml"
        config_file.write_text("""
plugins:
  - name: git-plugin
    source: git
    url: https://github.com/user/plugin.git
    ref: main
    enabled: true
""")

        config = PluginConfig.from_file(config_file)
        plugin_spec = config.plugins[0]

        installer = PackageInstaller()

        with patch.object(installer, "_run_pip") as mock_pip:
            mock_pip.return_value = (True, "Successfully installed")

            success, message = installer.install_from_spec(plugin_spec)

            assert success is True

    def test_install_from_local_spec(self, temp_dir: Path):
        """Test installing from a local PluginSpec."""
        from vllm_plugin_manager.config import PluginConfig
        from vllm_plugin_manager.sources.installer import PackageInstaller

        plugin_dir = temp_dir / "local-plugin"
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
        plugin_spec = config.plugins[0]

        installer = PackageInstaller()

        with patch.object(installer, "_run_pip") as mock_pip:
            mock_pip.return_value = (True, "Successfully installed")

            success, message = installer.install_from_spec(plugin_spec)

            assert success is True

    def test_install_from_unknown_source(self, temp_dir: Path):
        """Test that unknown source type raises error."""
        from vllm_plugin_manager.config import PluginSpec
        from vllm_plugin_manager.sources.installer import PackageInstaller, InstallerError

        spec = PluginSpec(
            name="unknown-plugin",
            source="unknown",
            enabled=True,
        )

        installer = PackageInstaller()

        with pytest.raises(InstallerError):
            installer.install_from_spec(spec)
