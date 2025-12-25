"""vLLM Plugin Manager - Dynamic plugin installer and loader for vLLM."""

import logging
import multiprocessing
import os
from pathlib import Path
from typing import Optional

__version__ = "0.1.0"

logger = logging.getLogger(__name__)

# Flag to ensure register() only runs once
_registered = False


def is_main_process() -> bool:
    """Check if we're running in the main process (not a worker)."""
    current = multiprocessing.current_process()
    return current.name == "MainProcess"


def register() -> None:
    """
    Entry point called by vLLM during startup.

    This function:
    1. Loads plugin configuration from YAML file
    2. Installs any missing plugins via pip
    3. Invalidates importlib cache so vLLM discovers new plugins

    Only runs in the main process and only once per session.
    """
    global _registered

    # Only run once
    if _registered:
        return
    _registered = True

    # Only run in main process (not worker processes)
    if not is_main_process():
        logger.debug("Skipping plugin manager in worker process")
        return

    # Import here to avoid circular imports
    from .config import get_config_path
    from .manager import PluginManager
    from .core.registry import get_registry_dir

    try:
        config_path = get_config_path()
        if config_path is None:
            logger.debug("No plugin config file found, skipping plugin installation")
            return

        logger.info(f"Loading plugin configuration from {config_path}")

        manager = PluginManager(
            config_path=config_path,
            registry_dir=get_registry_dir(),
        )

        results = manager.install_plugins()

        # Log results
        for plugin_id, (success, message) in results.items():
            if success:
                logger.info(f"Plugin '{plugin_id}': {message}")
            else:
                logger.error(f"Plugin '{plugin_id}' failed: {message}")

    except Exception as e:
        # Never crash vLLM due to plugin manager errors
        logger.error(f"Plugin manager error: {e}", exc_info=True)


# Re-export key classes for convenience
from .config import PluginConfig, PluginSpec, ConfigError, get_config_path
from .manager import PluginManager
from .core.registry import PluginRegistry, PluginStatus, get_registry_dir
from .core.discovery import EntryPointDiscovery
from .sources.installer import PackageInstaller, InstallerError

__all__ = [
    "register",
    "is_main_process",
    "PluginConfig",
    "PluginSpec",
    "ConfigError",
    "get_config_path",
    "PluginManager",
    "PluginRegistry",
    "PluginStatus",
    "get_registry_dir",
    "EntryPointDiscovery",
    "PackageInstaller",
    "InstallerError",
]
