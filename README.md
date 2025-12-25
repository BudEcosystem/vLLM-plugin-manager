# vLLM Plugin Manager

A manager to dynamically install and load vLLM plugins at startup without rebuilding images.

## Key Features

- **Automatic Plugin Installation**: Install plugins from PyPI, Git, or local directories on vLLM startup
- **Configuration-Driven**: Define plugins in a simple YAML config file
- **Zero Code Changes**: Works via vLLM's entry point system - no vLLM modifications needed
- **Multi-Source Support**: PyPI packages, Git repositories (with branch/tag/subdirectory), local paths
- **Persistent Registry**: Track installed plugins across restarts
- **Cache Invalidation**: Automatically refreshes Python's import cache for newly installed packages

## Installation

```bash
pip install vllm-plugin-manager
```

Or install from source:

```bash
git clone https://github.com/your-org/vllm-plugin-manager.git
cd vllm-plugin-manager
pip install -e .
```

## Quick Start

1. Create a plugin configuration file at `~/.config/vllm/plugins.yaml`:

```yaml
plugins:
  - name: vllm-entropy-decoder
    source: pypi
    package: vllm-entropy-decoder
    version: ">=0.1.0"
    enabled: true
```

2. Start vLLM normally - plugins will be installed automatically:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct
```

## Configuration

### Config File Location

The plugin manager looks for configuration in this order:

1. `$VLLM_PLUGIN_CONFIG` environment variable
2. `~/.config/vllm/plugins.yaml`

### Plugin Sources

#### PyPI Package

```yaml
plugins:
  - name: my-plugin
    source: pypi
    package: vllm-my-plugin      # PyPI package name
    version: ">=1.0.0"           # Optional: version specifier
    enabled: true
```

#### Git Repository

```yaml
plugins:
  - name: my-plugin
    source: git
    url: https://github.com/user/vllm-plugin.git
    ref: main                    # Optional: branch, tag, or commit
    subdirectory: plugins/foo    # Optional: for monorepos
    enabled: true
```

#### Local Directory

```yaml
plugins:
  - name: my-plugin
    source: local
    path: /path/to/my-plugin
    editable: true               # Optional: install in editable mode
    enabled: true
```

### Full Example

```yaml
plugins:
  # Production plugin from PyPI
  - name: vllm-entropy-decoder
    source: pypi
    package: vllm-entropy-decoder
    version: ">=0.1.0"
    enabled: true

  # Custom plugin from private Git repo
  - name: custom-decoder
    source: git
    url: https://github.com/myorg/custom-decoder.git
    ref: v2.0.0
    enabled: true

  # Development plugin (local)
  - name: dev-plugin
    source: local
    path: ~/projects/my-vllm-plugin
    editable: true
    enabled: true

  # Disabled plugin (won't be installed)
  - name: experimental-plugin
    source: pypi
    package: vllm-experimental
    enabled: false
```

## Docker Usage

### Build and Run

```bash
# Build the image
docker build -t vllm-with-plugins .

# Run with your plugin config
docker run --gpus all \
  -v ./my-plugins.yaml:/root/.config/vllm/plugins.yaml:ro \
  -p 8000:8000 \
  vllm-with-plugins \
  --model meta-llama/Llama-3.1-8B-Instruct
```

### Custom Base Image

```bash
# Use your own vLLM image as base
docker build --build-arg BASE_IMAGE=my-vllm:latest -t vllm-with-plugins .
```

### Using Docker Compose

```bash
# Copy and customize the example config
cp examples/plugins.yaml my-plugins.yaml

# Start the service
docker-compose -f examples/docker-compose.yaml up
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `VLLM_PLUGIN_CONFIG` | Path to plugins.yaml config file | `~/.config/vllm/plugins.yaml` |
| `VLLM_PLUGIN_REGISTRY_DIR` | Directory for plugin registry | `~/.local/share/vllm-plugins` |


## Plugin Registry

Installed plugins are tracked in `~/.local/share/vllm-plugins/registry.json`:

```json
{
  "version": "1.0",
  "plugins": {
    "vllm-entropy-decoder": {
      "name": "vllm-entropy-decoder",
      "source": "pypi",
      "package": "vllm-entropy-decoder",
      "version": "0.1.0",
      "status": "installed",
      "entry_points": ["vllm.logits_processors:entropy"]
    }
  }
}
```

## Development

```bash
# Clone the repository
git clone https://github.com/your-org/vllm-plugin-manager.git
cd vllm-plugin-manager

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest -v
```

## License

Apache-2.0
