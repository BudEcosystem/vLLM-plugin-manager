# vLLM with Plugin Manager
# This Dockerfile extends the official vLLM image with the plugin manager

ARG VLLM_VERSION=latest
FROM vllm/vllm-openai:${VLLM_VERSION}

# Install the plugin manager
COPY . /tmp/vllm-plugin-manager
RUN pip install /tmp/vllm-plugin-manager && \
    rm -rf /tmp/vllm-plugin-manager

# Create config directory
RUN mkdir -p /root/.config/vllm

# Copy default plugin configuration (can be overridden via volume mount)
COPY examples/plugins.yaml /root/.config/vllm/plugins.yaml

# Environment variables for plugin manager
# Override these as needed
ENV VLLM_PLUGIN_CONFIG=/root/.config/vllm/plugins.yaml
ENV VLLM_PLUGIN_REGISTRY_DIR=/root/.local/share/vllm-plugins

# The plugin manager automatically runs when vLLM starts
# via the vllm.general_plugins entry point

# Default command (same as base vLLM image)
ENTRYPOINT ["python", "-m", "vllm.entrypoints.openai.api_server"]
