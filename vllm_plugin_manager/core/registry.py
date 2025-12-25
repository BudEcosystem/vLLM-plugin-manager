"""Plugin registry for tracking installed plugins."""

import json
import logging
import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from filelock import FileLock

logger = logging.getLogger(__name__)


class PluginStatus(Enum):
    """Status of a plugin in the registry."""

    PENDING = "pending"
    INSTALLING = "installing"
    INSTALLED = "installed"
    FAILED = "failed"


def get_registry_dir() -> Path:
    """
    Get the directory for plugin registry storage.

    Priority:
    1. VLLM_PLUGIN_REGISTRY_DIR environment variable
    2. ~/.local/share/vllm-plugins/
    """
    env_dir = os.environ.get("VLLM_PLUGIN_REGISTRY_DIR")
    if env_dir:
        return Path(env_dir)

    home = Path.home()
    return home / ".local" / "share" / "vllm-plugins"


class PluginRegistry:
    """
    Persistent registry for tracking installed plugins.

    Uses JSON file storage with file locking for multi-process safety.
    """

    REGISTRY_VERSION = "1.0"

    def __init__(self, registry_dir: Optional[Path] = None):
        """
        Initialize the plugin registry.

        Args:
            registry_dir: Directory to store registry file. Defaults to get_registry_dir().
        """
        self.registry_dir = registry_dir or get_registry_dir()
        self.registry_dir.mkdir(parents=True, exist_ok=True)

        self.registry_file = self.registry_dir / "registry.json"
        self.lock_file = self.registry_dir / "registry.lock"
        self._lock = FileLock(self.lock_file)

        # Load or create registry
        self._data = self._load()

    def _load(self) -> Dict[str, Any]:
        """Load registry from file."""
        if not self.registry_file.exists():
            return self._create_empty_registry()

        try:
            with self._lock:
                with open(self.registry_file, "r") as f:
                    data = json.load(f)

                # Validate structure
                if not isinstance(data, dict) or "plugins" not in data:
                    logger.warning("Invalid registry file, creating new one")
                    return self._create_empty_registry()

                return data

        except json.JSONDecodeError as e:
            logger.warning(f"Corrupted registry file, creating new one: {e}")
            return self._create_empty_registry()

    def _create_empty_registry(self) -> Dict[str, Any]:
        """Create an empty registry structure."""
        data = {
            "version": self.REGISTRY_VERSION,
            "plugins": {},
        }
        self._save(data)
        return data

    def _save(self, data: Optional[Dict[str, Any]] = None) -> None:
        """Save registry to file."""
        if data is None:
            data = self._data

        with self._lock:
            with open(self.registry_file, "w") as f:
                json.dump(data, f, indent=2)

    def register_plugin(
        self,
        plugin_id: str,
        name: str,
        source: str,
        package: Optional[str] = None,
        version: Optional[str] = None,
        status: PluginStatus = PluginStatus.PENDING,
        entry_points: Optional[List[str]] = None,
        error: Optional[str] = None,
    ) -> None:
        """
        Register a new plugin or update existing one.

        Args:
            plugin_id: Unique identifier for the plugin
            name: Human-readable name
            source: Installation source (pypi, git, local)
            package: Package name (for PyPI)
            version: Installed version
            status: Current status
            entry_points: List of registered entry points
            error: Error message if failed
        """
        plugin_data = {
            "name": name,
            "source": source,
            "package": package,
            "version": version,
            "status": status.value if isinstance(status, PluginStatus) else status,
            "entry_points": entry_points or [],
        }

        if error:
            plugin_data["error"] = error

        self._data["plugins"][plugin_id] = plugin_data
        self._save()

        logger.debug(f"Registered plugin: {plugin_id}")

    def get_plugin(self, plugin_id: str) -> Optional[Dict[str, Any]]:
        """Get plugin data by ID."""
        return self._data["plugins"].get(plugin_id)

    def get_all_plugins(self) -> Dict[str, Dict[str, Any]]:
        """Get all registered plugins."""
        return self._data["plugins"].copy()

    def update_status(
        self,
        plugin_id: str,
        status: PluginStatus,
        error: Optional[str] = None,
    ) -> None:
        """Update plugin status."""
        if plugin_id not in self._data["plugins"]:
            logger.warning(f"Cannot update status: plugin '{plugin_id}' not found")
            return

        self._data["plugins"][plugin_id]["status"] = status.value
        if error:
            self._data["plugins"][plugin_id]["error"] = error
        elif "error" in self._data["plugins"][plugin_id] and status == PluginStatus.INSTALLED:
            # Clear error on successful install
            del self._data["plugins"][plugin_id]["error"]

        self._save()

    def update_entry_points(self, plugin_id: str, entry_points: List[str]) -> None:
        """Update entry points for a plugin."""
        if plugin_id not in self._data["plugins"]:
            logger.warning(f"Cannot update entry points: plugin '{plugin_id}' not found")
            return

        self._data["plugins"][plugin_id]["entry_points"] = entry_points
        self._save()

    def remove_plugin(self, plugin_id: str) -> None:
        """Remove a plugin from the registry."""
        if plugin_id in self._data["plugins"]:
            del self._data["plugins"][plugin_id]
            self._save()
            logger.debug(f"Removed plugin: {plugin_id}")

    def is_installed(self, plugin_id: str) -> bool:
        """Check if a plugin is installed."""
        plugin = self.get_plugin(plugin_id)
        if not plugin:
            return False
        return plugin.get("status") == PluginStatus.INSTALLED.value

    def get_plugins_by_status(self, status: PluginStatus) -> Dict[str, Dict[str, Any]]:
        """Get all plugins with a specific status."""
        return {
            pid: pdata
            for pid, pdata in self._data["plugins"].items()
            if pdata.get("status") == status.value
        }
