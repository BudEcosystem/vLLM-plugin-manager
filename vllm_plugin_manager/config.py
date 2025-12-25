"""Configuration loading and parsing for vLLM Plugin Manager."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


class ConfigError(Exception):
    """Raised when configuration is invalid."""

    pass


@dataclass
class PluginSpec:
    """Specification for a single plugin."""

    name: str
    source: str  # "pypi", "git", or "local"
    enabled: bool = True

    # PyPI source fields
    package: Optional[str] = None
    version: Optional[str] = None

    # Git source fields
    url: Optional[str] = None
    ref: Optional[str] = None
    subdirectory: Optional[str] = None

    # Local source fields
    path: Optional[str] = None
    editable: bool = True

    @property
    def plugin_id(self) -> str:
        """Get unique identifier for this plugin."""
        if self.source == "pypi" and self.package:
            return self.package
        return self.name

    def get_install_spec(self) -> str:
        """Get pip install specification string."""
        if self.source == "pypi":
            if self.version:
                return f"{self.package}{self.version}"
            return self.package or self.name

        elif self.source == "git":
            spec = f"git+{self.url}"
            if self.ref:
                spec += f"@{self.ref}"
            if self.subdirectory:
                spec += f"#subdirectory={self.subdirectory}"
            return spec

        elif self.source == "local":
            return str(self.path) if self.path else ""

        else:
            raise ConfigError(f"Unknown source type: {self.source}")

    def validate(self) -> None:
        """Validate the plugin specification."""
        if not self.name:
            raise ConfigError("Plugin 'name' is required")

        if not self.source:
            raise ConfigError(f"Plugin '{self.name}' missing 'source' field")

        if self.source == "pypi":
            if not self.package:
                raise ConfigError(f"PyPI plugin '{self.name}' missing 'package' field")

        elif self.source == "git":
            if not self.url:
                raise ConfigError(f"Git plugin '{self.name}' missing 'url' field")

        elif self.source == "local":
            if not self.path:
                raise ConfigError(f"Local plugin '{self.name}' missing 'path' field")

        elif self.source not in ("pypi", "git", "local"):
            raise ConfigError(f"Plugin '{self.name}' has unknown source: {self.source}")


@dataclass
class PluginConfig:
    """Configuration containing list of plugins to install."""

    plugins: List[PluginSpec] = field(default_factory=list)

    @classmethod
    def from_file(cls, path: Path) -> "PluginConfig":
        """Load configuration from a YAML file."""
        try:
            with open(path, "r") as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigError(f"Invalid YAML in {path}: {e}")
        except FileNotFoundError:
            raise ConfigError(f"Config file not found: {path}")

        if data is None:
            data = {}

        plugins_data = data.get("plugins", [])
        plugins = []

        for plugin_data in plugins_data:
            if not isinstance(plugin_data, dict):
                raise ConfigError(f"Invalid plugin entry: {plugin_data}")

            spec = PluginSpec(
                name=plugin_data.get("name", ""),
                source=plugin_data.get("source", ""),
                enabled=plugin_data.get("enabled", True),
                package=plugin_data.get("package"),
                version=plugin_data.get("version"),
                url=plugin_data.get("url"),
                ref=plugin_data.get("ref"),
                subdirectory=plugin_data.get("subdirectory"),
                path=plugin_data.get("path"),
                editable=plugin_data.get("editable", True),
            )

            # Validate the spec
            spec.validate()
            plugins.append(spec)

        return cls(plugins=plugins)

    def get_enabled_plugins(self) -> List[PluginSpec]:
        """Get only the enabled plugins."""
        return [p for p in self.plugins if p.enabled]


def get_config_path() -> Optional[Path]:
    """
    Get the path to the plugin configuration file.

    Priority:
    1. VLLM_PLUGIN_CONFIG environment variable
    2. ~/.config/vllm/plugins.yaml

    Returns None if no config file exists.
    """
    # Check environment variable first
    env_path = os.environ.get("VLLM_PLUGIN_CONFIG")
    if env_path:
        path = Path(env_path)
        if path.exists():
            return path
        # If explicitly set but doesn't exist, return None
        # (user may not have mounted a config file yet)
        return None

    # Check default location
    home = Path.home()
    default_path = home / ".config" / "vllm" / "plugins.yaml"
    if default_path.exists():
        return default_path

    return None
