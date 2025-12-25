# vLLM with Plugin Manager
#
# Build:
#   docker build -t vllm-with-plugins .
#
# With custom base image:
#   docker build --build-arg BASE_IMAGE=my-vllm:latest -t vllm-with-plugins .
#
# Run:
#   docker run --gpus all \
#     -v ./plugins.yaml:/root/.config/vllm/plugins.yaml:ro \
#     -p 8000:8000 \
#     vllm-with-plugins --model meta-llama/Llama-3.1-8B-Instruct

ARG BASE_IMAGE=vllm/vllm-openai:latest
FROM ${BASE_IMAGE}

# Install plugin manager
COPY vllm_plugin_manager /opt/vllm-plugin-manager/vllm_plugin_manager
COPY pyproject.toml /opt/vllm-plugin-manager/
RUN pip install --no-cache-dir /opt/vllm-plugin-manager

# Config directory (mount your plugins.yaml here)
RUN mkdir -p /root/.config/vllm

# Environment variables
ENV VLLM_PLUGIN_CONFIG=/root/.config/vllm/plugins.yaml
ENV VLLM_PLUGIN_REGISTRY_DIR=/root/.local/share/vllm-plugins
