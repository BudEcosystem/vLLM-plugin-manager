"""Plugin Manager - Orchestrates plugin installation and lifecycle."""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

from .config import PluginConfig, PluginSpec
from .core.registry import PluginRegistry, PluginStatus
from .core.discovery import EntryPointDiscovery
from .sources.installer import PackageInstaller

logger = logging.getLogger(__name__)


class PluginManager:
    """
    Main orchestrator for plugin installation.

    Coordinates:
    - Loading plugin configuration from YAML
    - Installing plugins via pip
    - Tracking installed plugins in registry
    - Invalidating caches for entry point discovery
    """

    def __init__(
        self,
        config_path: Path,
        registry_dir: Path,
    ):
        """
        Initialize the plugin manager.

        Args:
            config_path: Path to plugins.yaml configuration file
            registry_dir: Directory for plugin registry storage
        """
        self.config_path = config_path
        self.config = PluginConfig.from_file(config_path)

        self.registry = PluginRegistry(registry_dir=registry_dir)
        self.discovery = EntryPointDiscovery()
        self.installer = PackageInstaller()

    def install_plugins(self) -> Dict[str, Tuple[bool, str]]:
        """
        Install all enabled plugins from configuration.

        Returns:
            Dict mapping plugin_id to (success, message) tuple
        """
        results = {}

        # Get only enabled plugins
        enabled_plugins = self.config.get_enabled_plugins()

        if not enabled_plugins:
            logger.info("No enabled plugins to install")
            return results

        logger.info(f"Installing {len(enabled_plugins)} plugin(s)")

        # Take snapshot before installation for diff detection
        self.discovery.take_snapshot()

        installed_any = False

        for spec in enabled_plugins:
            plugin_id = spec.plugin_id

            # Skip if already installed
            if self.registry.is_installed(plugin_id):
                logger.info(f"Plugin '{plugin_id}' already installed, skipping")
                results[plugin_id] = (True, "Already installed")
                continue

            # Register as pending
            self.registry.register_plugin(
                plugin_id=plugin_id,
                name=spec.name,
                source=spec.source,
                package=spec.package,
                status=PluginStatus.PENDING,
            )

            # Update to installing
            self.registry.update_status(plugin_id, PluginStatus.INSTALLING)

            try:
                # Install the plugin
                success, message = self.installer.install_from_spec(spec)

                if success:
                    # Get installed version
                    version = self.installer.get_installed_version(spec.package or spec.name)

                    self.registry.register_plugin(
                        plugin_id=plugin_id,
                        name=spec.name,
                        source=spec.source,
                        package=spec.package,
                        version=version,
                        status=PluginStatus.INSTALLED,
                    )

                    installed_any = True
                    results[plugin_id] = (True, f"Installed {version or 'successfully'}")
                    logger.info(f"Successfully installed plugin '{plugin_id}'")

                else:
                    self.registry.update_status(
                        plugin_id,
                        PluginStatus.FAILED,
                        error=message,
                    )
                    results[plugin_id] = (False, message)
                    logger.error(f"Failed to install plugin '{plugin_id}': {message}")

            except Exception as e:
                error_msg = str(e)
                self.registry.update_status(
                    plugin_id,
                    PluginStatus.FAILED,
                    error=error_msg,
                )
                results[plugin_id] = (False, error_msg)
                logger.error(f"Error installing plugin '{plugin_id}': {e}")

        # Invalidate cache if any plugins were installed
        if installed_any:
            logger.info("Invalidating entry point cache")
            self.discovery.invalidate_cache()

            # Update registry with discovered entry points
            self._update_entry_points()

        return results

    def _update_entry_points(self) -> None:
        """Update registry with newly discovered entry points."""
        new_eps = self.discovery.get_new_entry_points()

        for group, eps in new_eps.items():
            for ep in eps:
                # Try to find which plugin this belongs to
                try:
                    if hasattr(ep, "dist") and ep.dist:
                        pkg_name = ep.dist.name
                        plugin = self.registry.get_plugin(pkg_name)

                        if plugin:
                            current_eps = plugin.get("entry_points", [])
                            ep_str = f"{group}:{ep.name}"
                            if ep_str not in current_eps:
                                current_eps.append(ep_str)
                                self.registry.update_entry_points(pkg_name, current_eps)

                except Exception as e:
                    logger.debug(f"Could not update entry points: {e}")

    def get_installed_plugins(self) -> List[Dict]:
        """
        Get list of installed plugins.

        Returns:
            List of plugin data dictionaries
        """
        installed = self.registry.get_plugins_by_status(PluginStatus.INSTALLED)
        return list(installed.values())

    def get_plugin_status(self, plugin_id: str) -> Dict:
        """
        Get status of a specific plugin.

        Args:
            plugin_id: Plugin identifier

        Returns:
            Plugin data dictionary or empty dict if not found
        """
        plugin = self.registry.get_plugin(plugin_id)
        return plugin or {}
