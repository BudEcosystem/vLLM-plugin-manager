"""Sample vLLM Plugin - Demonstrates plugin structure."""

import logging

logger = logging.getLogger(__name__)

# Track if already registered (important for multi-process safety)
_registered = False


def register() -> None:
    """
    Entry point called by vLLM via vllm.general_plugins.

    Use this to:
    - Register custom model architectures
    - Apply patches to vLLM internals
    - Initialize plugin-wide state
    """
    global _registered

    if _registered:
        return
    _registered = True

    logger.info("Sample plugin registered successfully!")

    # Example: Register a custom model architecture
    # from vllm import ModelRegistry
    # from .models import MyCustomModel
    # ModelRegistry.register_model("MyCustomModel", MyCustomModel)
