"""Sample Logits Processor for vLLM."""

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.sample.sampler import SamplingParams


class SampleLogitsProcessor:
    """
    Sample logits processor that demonstrates the vLLM v1 interface.

    This processor applies a simple temperature scaling to logits.
    In vLLM v1, logits processors apply GLOBALLY to all requests.

    Entry point registration in pyproject.toml:
        [project.entry-points."vllm.logits_processors"]
        sample_processor = "my_plugin.processor:SampleLogitsProcessor"
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        device: torch.device,
        is_pin_memory: bool,
    ):
        """
        Initialize the logits processor.

        Args:
            vllm_config: vLLM configuration object
            device: Target device (cuda, cpu, etc.)
            is_pin_memory: Whether to use pinned memory
        """
        self.device = device
        self.temperature_scale = 1.0  # Could be loaded from config

    def is_argmax_invariant(self) -> bool:
        """
        Return True if this processor preserves argmax ordering.

        If True, greedy decoding can skip the full softmax computation.
        Temperature scaling with scale=1.0 is argmax invariant.
        """
        return self.temperature_scale == 1.0

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply the logits processing.

        Args:
            logits: Raw logits tensor of shape [batch_size, vocab_size]

        Returns:
            Processed logits tensor of same shape
        """
        if self.temperature_scale != 1.0:
            logits = logits / self.temperature_scale
        return logits

    @classmethod
    def validate_params(cls, sampling_params: "SamplingParams") -> None:
        """
        Validate that sampling parameters are compatible with this processor.

        Raise ValueError if incompatible parameters are detected.

        Args:
            sampling_params: The sampling parameters to validate
        """
        # Example: Check for incompatible settings
        # if sampling_params.some_param:
        #     raise ValueError("This processor doesn't support some_param")
        pass
