"""Pytest configuration and shared fixtures."""

import os
import tempfile
from pathlib import Path
from typing import Generator

import pytest


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_plugins_yaml(temp_dir: Path) -> Path:
    """Create a sample plugins.yaml config file."""
    config_file = temp_dir / "plugins.yaml"
    config_file.write_text("""
plugins:
  - name: vllm-entropy-decoder
    source: pypi
    package: vllm-entropy-decoder
    version: ">=0.1.0"
    enabled: true

  - name: my-custom-plugin
    source: git
    url: https://github.com/user/my-plugin.git
    ref: main
    enabled: true

  - name: disabled-plugin
    source: pypi
    package: some-plugin
    enabled: false
""")
    return config_file


@pytest.fixture
def sample_registry_json(temp_dir: Path) -> Path:
    """Create a sample registry.json file."""
    registry_file = temp_dir / "registry.json"
    registry_file.write_text("""
{
  "plugins": {
    "vllm-entropy-decoder": {
      "name": "vllm-entropy-decoder",
      "source": "pypi",
      "package": "vllm-entropy-decoder",
      "version": "0.1.0",
      "status": "installed",
      "entry_points": ["vllm.logits_processors:entropy"]
    }
  },
  "version": "1.0"
}
""")
    return registry_file


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear plugin manager environment variables."""
    env_vars = [
        "VLLM_PLUGIN_CONFIG",
        "VLLM_PLUGIN_REGISTRY_DIR",
        "VLLM_PLUGIN_AUTO_INSTALL",
        "VLLM_PLUGIN_CACHE_DIR",
    ]
    for var in env_vars:
        monkeypatch.delenv(var, raising=False)
