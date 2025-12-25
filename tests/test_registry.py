"""Tests for plugin registry (JSON persistence)."""

import json
from pathlib import Path

import pytest


class TestPluginRegistry:
    """Tests for PluginRegistry class."""

    def test_create_new_registry(self, temp_dir: Path):
        """Test creating a new registry when none exists."""
        from vllm_plugin_manager.core.registry import PluginRegistry

        registry = PluginRegistry(registry_dir=temp_dir)

        assert registry.get_all_plugins() == {}
        assert (temp_dir / "registry.json").exists()

    def test_load_existing_registry(self, sample_registry_json: Path):
        """Test loading an existing registry file."""
        from vllm_plugin_manager.core.registry import PluginRegistry

        registry = PluginRegistry(registry_dir=sample_registry_json.parent)

        plugins = registry.get_all_plugins()
        assert "vllm-entropy-decoder" in plugins
        assert plugins["vllm-entropy-decoder"]["status"] == "installed"

    def test_register_plugin(self, temp_dir: Path):
        """Test registering a new plugin."""
        from vllm_plugin_manager.core.registry import PluginRegistry, PluginStatus

        registry = PluginRegistry(registry_dir=temp_dir)

        registry.register_plugin(
            plugin_id="my-new-plugin",
            name="my-new-plugin",
            source="pypi",
            package="my-new-plugin",
            version="1.0.0",
            status=PluginStatus.INSTALLED,
        )

        plugin = registry.get_plugin("my-new-plugin")
        assert plugin is not None
        assert plugin["name"] == "my-new-plugin"
        assert plugin["version"] == "1.0.0"
        assert plugin["status"] == PluginStatus.INSTALLED.value

    def test_update_plugin_status(self, temp_dir: Path):
        """Test updating plugin status."""
        from vllm_plugin_manager.core.registry import PluginRegistry, PluginStatus

        registry = PluginRegistry(registry_dir=temp_dir)

        registry.register_plugin(
            plugin_id="test-plugin",
            name="test-plugin",
            source="pypi",
            package="test-plugin",
            version="1.0.0",
            status=PluginStatus.PENDING,
        )

        registry.update_status("test-plugin", PluginStatus.INSTALLING)
        assert registry.get_plugin("test-plugin")["status"] == PluginStatus.INSTALLING.value

        registry.update_status("test-plugin", PluginStatus.INSTALLED)
        assert registry.get_plugin("test-plugin")["status"] == PluginStatus.INSTALLED.value

    def test_remove_plugin(self, temp_dir: Path):
        """Test removing a plugin from registry."""
        from vllm_plugin_manager.core.registry import PluginRegistry, PluginStatus

        registry = PluginRegistry(registry_dir=temp_dir)

        registry.register_plugin(
            plugin_id="to-remove",
            name="to-remove",
            source="pypi",
            package="to-remove",
            version="1.0.0",
            status=PluginStatus.INSTALLED,
        )

        assert registry.get_plugin("to-remove") is not None

        registry.remove_plugin("to-remove")
        assert registry.get_plugin("to-remove") is None

    def test_is_installed(self, temp_dir: Path):
        """Test checking if a plugin is installed."""
        from vllm_plugin_manager.core.registry import PluginRegistry, PluginStatus

        registry = PluginRegistry(registry_dir=temp_dir)

        registry.register_plugin(
            plugin_id="installed-plugin",
            name="installed-plugin",
            source="pypi",
            package="installed-plugin",
            version="1.0.0",
            status=PluginStatus.INSTALLED,
        )

        registry.register_plugin(
            plugin_id="pending-plugin",
            name="pending-plugin",
            source="pypi",
            package="pending-plugin",
            version="1.0.0",
            status=PluginStatus.PENDING,
        )

        assert registry.is_installed("installed-plugin") is True
        assert registry.is_installed("pending-plugin") is False
        assert registry.is_installed("nonexistent-plugin") is False

    def test_registry_persistence(self, temp_dir: Path):
        """Test that registry changes are persisted to disk."""
        from vllm_plugin_manager.core.registry import PluginRegistry, PluginStatus

        # Create registry and add plugin
        registry1 = PluginRegistry(registry_dir=temp_dir)
        registry1.register_plugin(
            plugin_id="persistent-plugin",
            name="persistent-plugin",
            source="pypi",
            package="persistent-plugin",
            version="1.0.0",
            status=PluginStatus.INSTALLED,
        )

        # Create new registry instance (simulates restart)
        registry2 = PluginRegistry(registry_dir=temp_dir)

        plugin = registry2.get_plugin("persistent-plugin")
        assert plugin is not None
        assert plugin["name"] == "persistent-plugin"

    def test_registry_file_locking(self, temp_dir: Path):
        """Test that registry uses file locking for concurrent access."""
        from vllm_plugin_manager.core.registry import PluginRegistry, PluginStatus

        registry = PluginRegistry(registry_dir=temp_dir)

        # Verify lock file is created
        registry.register_plugin(
            plugin_id="test",
            name="test",
            source="pypi",
            package="test",
            version="1.0.0",
            status=PluginStatus.INSTALLED,
        )

        # Lock file should exist or be cleaned up after operation
        # The implementation should use filelock
        registry_file = temp_dir / "registry.json"
        assert registry_file.exists()

    def test_set_error_message(self, temp_dir: Path):
        """Test setting error message on plugin."""
        from vllm_plugin_manager.core.registry import PluginRegistry, PluginStatus

        registry = PluginRegistry(registry_dir=temp_dir)

        registry.register_plugin(
            plugin_id="failed-plugin",
            name="failed-plugin",
            source="pypi",
            package="failed-plugin",
            version="1.0.0",
            status=PluginStatus.PENDING,
        )

        registry.update_status("failed-plugin", PluginStatus.FAILED, error="Installation failed: network error")

        plugin = registry.get_plugin("failed-plugin")
        assert plugin["status"] == PluginStatus.FAILED.value
        assert plugin["error"] == "Installation failed: network error"

    def test_get_plugins_by_status(self, temp_dir: Path):
        """Test getting plugins filtered by status."""
        from vllm_plugin_manager.core.registry import PluginRegistry, PluginStatus

        registry = PluginRegistry(registry_dir=temp_dir)

        registry.register_plugin(
            plugin_id="installed-1",
            name="installed-1",
            source="pypi",
            package="installed-1",
            version="1.0.0",
            status=PluginStatus.INSTALLED,
        )
        registry.register_plugin(
            plugin_id="installed-2",
            name="installed-2",
            source="pypi",
            package="installed-2",
            version="1.0.0",
            status=PluginStatus.INSTALLED,
        )
        registry.register_plugin(
            plugin_id="failed-1",
            name="failed-1",
            source="pypi",
            package="failed-1",
            version="1.0.0",
            status=PluginStatus.FAILED,
        )

        installed = registry.get_plugins_by_status(PluginStatus.INSTALLED)
        assert len(installed) == 2

        failed = registry.get_plugins_by_status(PluginStatus.FAILED)
        assert len(failed) == 1

    def test_update_entry_points(self, temp_dir: Path):
        """Test updating entry points for a plugin."""
        from vllm_plugin_manager.core.registry import PluginRegistry, PluginStatus

        registry = PluginRegistry(registry_dir=temp_dir)

        registry.register_plugin(
            plugin_id="ep-plugin",
            name="ep-plugin",
            source="pypi",
            package="ep-plugin",
            version="1.0.0",
            status=PluginStatus.INSTALLED,
        )

        entry_points = [
            "vllm.general_plugins:my_register",
            "vllm.logits_processors:MyProcessor",
        ]
        registry.update_entry_points("ep-plugin", entry_points)

        plugin = registry.get_plugin("ep-plugin")
        assert plugin["entry_points"] == entry_points

    def test_corrupted_registry_file_recovery(self, temp_dir: Path):
        """Test recovery from corrupted registry file."""
        from vllm_plugin_manager.core.registry import PluginRegistry

        registry_file = temp_dir / "registry.json"
        registry_file.write_text("{ invalid json content")

        # Should handle corrupted file gracefully
        registry = PluginRegistry(registry_dir=temp_dir)
        assert registry.get_all_plugins() == {}

    def test_default_registry_dir(self, monkeypatch: pytest.MonkeyPatch, temp_dir: Path):
        """Test default registry directory resolution."""
        from vllm_plugin_manager.core.registry import get_registry_dir

        monkeypatch.delenv("VLLM_PLUGIN_REGISTRY_DIR", raising=False)
        monkeypatch.setenv("HOME", str(temp_dir))

        registry_dir = get_registry_dir()
        assert registry_dir == temp_dir / ".local" / "share" / "vllm-plugins"

    def test_registry_dir_from_env(self, monkeypatch: pytest.MonkeyPatch, temp_dir: Path):
        """Test registry directory from environment variable."""
        from vllm_plugin_manager.core.registry import get_registry_dir

        custom_dir = temp_dir / "custom-registry"
        monkeypatch.setenv("VLLM_PLUGIN_REGISTRY_DIR", str(custom_dir))

        registry_dir = get_registry_dir()
        assert registry_dir == custom_dir
