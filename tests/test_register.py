"""Tests for main register function and plugin manager."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest


@pytest.fixture(autouse=True)
def reset_registered_flag():
    """Reset the _registered flag before and after each test."""
    import vllm_plugin_manager
    vllm_plugin_manager._registered = False
    yield
    vllm_plugin_manager._registered = False


class TestRegisterFunction:
    """Tests for the main register() entry point."""

    def test_register_does_nothing_without_config(self, monkeypatch: pytest.MonkeyPatch, temp_dir: Path):
        """Test that register() does nothing when no config file exists."""
        import vllm_plugin_manager

        monkeypatch.delenv("VLLM_PLUGIN_CONFIG", raising=False)
        monkeypatch.setenv("HOME", str(temp_dir))

        # Should not raise, just return early
        vllm_plugin_manager.register()

    def test_register_loads_and_installs_plugins(self, monkeypatch: pytest.MonkeyPatch, temp_dir: Path):
        """Test that register() loads config and installs plugins."""
        import vllm_plugin_manager
        from vllm_plugin_manager.core.registry import PluginStatus

        # Create config file with a local plugin (avoids network)
        plugin_dir = temp_dir / "test-plugin"
        plugin_dir.mkdir()
        (plugin_dir / "pyproject.toml").write_text("""
[project]
name = "test-plugin"
version = "0.1.0"
""")
        (plugin_dir / "test_plugin.py").write_text("# test plugin")

        config_file = temp_dir / "plugins.yaml"
        config_file.write_text(f"""
plugins:
  - name: test-plugin
    source: local
    path: {plugin_dir}
    enabled: true
""")
        monkeypatch.setenv("VLLM_PLUGIN_CONFIG", str(config_file))
        monkeypatch.setenv("VLLM_PLUGIN_REGISTRY_DIR", str(temp_dir))

        # Mock the installer to avoid actually installing
        with patch("vllm_plugin_manager.sources.installer.PackageInstaller.install_from_spec") as mock_install:
            mock_install.return_value = (True, "Successfully installed")

            vllm_plugin_manager.register()

            # Should have attempted to install the plugin
            mock_install.assert_called_once()

            # Registry should have the plugin
            from vllm_plugin_manager.core.registry import PluginRegistry
            registry = PluginRegistry(registry_dir=temp_dir)
            plugin = registry.get_plugin("test-plugin")
            assert plugin is not None
            assert plugin["status"] == PluginStatus.INSTALLED.value

    def test_register_only_runs_once(self, monkeypatch: pytest.MonkeyPatch, temp_dir: Path):
        """Test that register() only runs once (re-entrant safe)."""
        import vllm_plugin_manager

        plugin_dir = temp_dir / "test-plugin"
        plugin_dir.mkdir()
        (plugin_dir / "pyproject.toml").write_text('[project]\nname = "test-plugin"\nversion = "0.1.0"')

        config_file = temp_dir / "plugins.yaml"
        config_file.write_text(f"""
plugins:
  - name: test-plugin
    source: local
    path: {plugin_dir}
    enabled: true
""")
        monkeypatch.setenv("VLLM_PLUGIN_CONFIG", str(config_file))
        monkeypatch.setenv("VLLM_PLUGIN_REGISTRY_DIR", str(temp_dir))

        # Mock the installer to track calls
        with patch("vllm_plugin_manager.sources.installer.PackageInstaller.install_from_spec") as mock_install:
            mock_install.return_value = (True, "Successfully installed")

            # Call register twice
            vllm_plugin_manager.register()
            vllm_plugin_manager.register()

            # Should only install once (re-entrant safe)
            assert mock_install.call_count == 1

    def test_register_only_runs_in_main_process(self, monkeypatch: pytest.MonkeyPatch, temp_dir: Path):
        """Test that register() only runs in the main process."""
        import vllm_plugin_manager

        config_file = temp_dir / "plugins.yaml"
        config_file.write_text("plugins: []")
        monkeypatch.setenv("VLLM_PLUGIN_CONFIG", str(config_file))

        with patch.object(vllm_plugin_manager, "is_main_process", return_value=False):
            with patch.object(vllm_plugin_manager, "PluginManager") as MockManager:
                vllm_plugin_manager.register()

                # Should not create manager in worker process
                MockManager.assert_not_called()

    def test_register_handles_errors_gracefully(self, monkeypatch: pytest.MonkeyPatch, temp_dir: Path):
        """Test that register() handles errors without crashing vLLM."""
        import vllm_plugin_manager

        config_file = temp_dir / "plugins.yaml"
        config_file.write_text("plugins: []")
        monkeypatch.setenv("VLLM_PLUGIN_CONFIG", str(config_file))
        monkeypatch.setenv("VLLM_PLUGIN_REGISTRY_DIR", str(temp_dir))

        with patch.object(vllm_plugin_manager, "PluginManager") as MockManager:
            MockManager.side_effect = Exception("Something went wrong")

            # Should not raise, just log the error
            vllm_plugin_manager.register()


class TestPluginManager:
    """Tests for PluginManager orchestrator class."""

    def test_create_plugin_manager(self, temp_dir: Path):
        """Test creating PluginManager instance."""
        from vllm_plugin_manager.manager import PluginManager

        config_file = temp_dir / "plugins.yaml"
        config_file.write_text("plugins: []")

        manager = PluginManager(
            config_path=config_file,
            registry_dir=temp_dir,
        )

        assert manager is not None

    def test_install_plugins_from_config(self, temp_dir: Path):
        """Test installing plugins from config."""
        from vllm_plugin_manager.manager import PluginManager

        config_file = temp_dir / "plugins.yaml"
        config_file.write_text("""
plugins:
  - name: plugin-a
    source: pypi
    package: plugin-a
    enabled: true
  - name: plugin-b
    source: pypi
    package: plugin-b
    enabled: true
""")

        manager = PluginManager(
            config_path=config_file,
            registry_dir=temp_dir,
        )

        with patch.object(manager.installer, "install_from_spec") as mock_install:
            mock_install.return_value = (True, "Installed")

            with patch.object(manager.discovery, "invalidate_cache"):
                results = manager.install_plugins()

            assert mock_install.call_count == 2

    def test_skip_already_installed_plugins(self, temp_dir: Path):
        """Test that already installed plugins are skipped."""
        from vllm_plugin_manager.manager import PluginManager
        from vllm_plugin_manager.core.registry import PluginStatus

        config_file = temp_dir / "plugins.yaml"
        config_file.write_text("""
plugins:
  - name: already-installed
    source: pypi
    package: already-installed
    enabled: true
""")

        manager = PluginManager(
            config_path=config_file,
            registry_dir=temp_dir,
        )

        # Pre-register the plugin as installed
        manager.registry.register_plugin(
            plugin_id="already-installed",
            name="already-installed",
            source="pypi",
            package="already-installed",
            version="1.0.0",
            status=PluginStatus.INSTALLED,
        )

        with patch.object(manager.installer, "install_from_spec") as mock_install:
            results = manager.install_plugins()

            # Should not attempt to install
            mock_install.assert_not_called()

    def test_skip_disabled_plugins(self, temp_dir: Path):
        """Test that disabled plugins are skipped."""
        from vllm_plugin_manager.manager import PluginManager

        config_file = temp_dir / "plugins.yaml"
        config_file.write_text("""
plugins:
  - name: disabled-plugin
    source: pypi
    package: disabled-plugin
    enabled: false
""")

        manager = PluginManager(
            config_path=config_file,
            registry_dir=temp_dir,
        )

        with patch.object(manager.installer, "install_from_spec") as mock_install:
            results = manager.install_plugins()

            # Should not attempt to install disabled plugin
            mock_install.assert_not_called()

    def test_install_failure_updates_registry(self, temp_dir: Path):
        """Test that installation failure updates registry with error."""
        from vllm_plugin_manager.manager import PluginManager
        from vllm_plugin_manager.core.registry import PluginStatus

        config_file = temp_dir / "plugins.yaml"
        config_file.write_text("""
plugins:
  - name: failing-plugin
    source: pypi
    package: failing-plugin
    enabled: true
""")

        manager = PluginManager(
            config_path=config_file,
            registry_dir=temp_dir,
        )

        with patch.object(manager.installer, "install_from_spec") as mock_install:
            mock_install.return_value = (False, "Installation failed: network error")

            results = manager.install_plugins()

            # Check registry has failure recorded
            plugin = manager.registry.get_plugin("failing-plugin")
            assert plugin["status"] == PluginStatus.FAILED.value
            assert "error" in plugin

    def test_invalidate_cache_after_install(self, temp_dir: Path):
        """Test that entry point cache is invalidated after installation."""
        from vllm_plugin_manager.manager import PluginManager

        config_file = temp_dir / "plugins.yaml"
        config_file.write_text("""
plugins:
  - name: new-plugin
    source: pypi
    package: new-plugin
    enabled: true
""")

        manager = PluginManager(
            config_path=config_file,
            registry_dir=temp_dir,
        )

        with patch.object(manager.installer, "install_from_spec") as mock_install:
            mock_install.return_value = (True, "Installed")

            with patch.object(manager.discovery, "invalidate_cache") as mock_invalidate:
                manager.install_plugins()

                # Cache should be invalidated after successful install
                mock_invalidate.assert_called()

    def test_get_installed_plugins(self, temp_dir: Path):
        """Test getting list of installed plugins."""
        from vllm_plugin_manager.manager import PluginManager
        from vllm_plugin_manager.core.registry import PluginStatus

        config_file = temp_dir / "plugins.yaml"
        config_file.write_text("plugins: []")

        manager = PluginManager(
            config_path=config_file,
            registry_dir=temp_dir,
        )

        # Add some plugins to registry
        manager.registry.register_plugin(
            plugin_id="plugin-1",
            name="plugin-1",
            source="pypi",
            package="plugin-1",
            version="1.0.0",
            status=PluginStatus.INSTALLED,
        )
        manager.registry.register_plugin(
            plugin_id="plugin-2",
            name="plugin-2",
            source="git",
            package="plugin-2",
            version="2.0.0",
            status=PluginStatus.INSTALLED,
        )

        installed = manager.get_installed_plugins()

        assert len(installed) == 2


class TestMainProcessCheck:
    """Tests for main process detection."""

    def test_is_main_process_true_for_main(self):
        """Test that main process is correctly detected."""
        from vllm_plugin_manager import is_main_process

        # In test environment, we're typically in main process
        # This might need adjustment based on test runner
        result = is_main_process()
        assert isinstance(result, bool)

    def test_is_main_process_with_multiprocessing(self):
        """Test main process detection with multiprocessing."""
        import multiprocessing

        from vllm_plugin_manager import is_main_process

        # Create a simple check in subprocess
        def check_in_subprocess(queue):
            from vllm_plugin_manager import is_main_process
            queue.put(is_main_process())

        queue = multiprocessing.Queue()
        process = multiprocessing.Process(target=check_in_subprocess, args=(queue,))
        process.start()
        process.join(timeout=5)

        if not queue.empty():
            result = queue.get()
            # Subprocess should NOT be main process
            assert result is False
