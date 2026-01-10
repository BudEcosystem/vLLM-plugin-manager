"""Version compatibility checking for vLLM plugins."""

import importlib.metadata
import logging
from typing import Optional, Tuple

from packaging.specifiers import SpecifierSet, InvalidSpecifier
from packaging.version import Version, InvalidVersion

logger = logging.getLogger(__name__)


def get_vllm_version() -> Optional[str]:
    """
    Get the currently installed vLLM version.

    Returns:
        Version string or None if vLLM is not installed
    """
    try:
        return importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError:
        logger.warning("vLLM package not found")
        return None


def parse_version(version_str: str) -> Optional[Version]:
    """
    Parse a version string into a Version object.

    Args:
        version_str: Version string (e.g., "0.6.0", "1.0.0.dev123")

    Returns:
        Version object or None if parsing fails
    """
    try:
        return Version(version_str)
    except InvalidVersion:
        logger.warning(f"Invalid version string: {version_str}")
        return None


def check_version_constraint(
    version: str,
    constraint: str,
) -> Tuple[bool, str]:
    """
    Check if a version satisfies a constraint.

    Args:
        version: Version string to check (e.g., "0.6.0")
        constraint: Version constraint/specifier (e.g., ">=0.5.0,<1.0.0")

    Returns:
        Tuple of (is_compatible, message)
    """
    try:
        specifier = SpecifierSet(constraint)
    except InvalidSpecifier as e:
        return False, f"Invalid version constraint '{constraint}': {e}"

    parsed_version = parse_version(version)
    if parsed_version is None:
        return False, f"Could not parse version '{version}'"

    if parsed_version in specifier:
        return True, f"Version {version} satisfies {constraint}"
    else:
        return False, f"Version {version} does not satisfy {constraint}"


def check_vllm_compatibility(constraint: str) -> Tuple[bool, str]:
    """
    Check if the installed vLLM version satisfies a constraint.

    Args:
        constraint: Version constraint (e.g., ">=0.6.0,<1.0.0")

    Returns:
        Tuple of (is_compatible, message)
    """
    vllm_version = get_vllm_version()

    if vllm_version is None:
        return False, "vLLM is not installed"

    return check_version_constraint(vllm_version, constraint)


class VersionChecker:
    """
    Utility class for checking plugin version compatibility.
    """

    def __init__(self):
        """Initialize the version checker."""
        self._vllm_version: Optional[str] = None
        self._vllm_version_checked = False

    @property
    def vllm_version(self) -> Optional[str]:
        """Get the cached vLLM version."""
        if not self._vllm_version_checked:
            self._vllm_version = get_vllm_version()
            self._vllm_version_checked = True
        return self._vllm_version

    def is_compatible(self, constraint: Optional[str]) -> Tuple[bool, str]:
        """
        Check if the installed vLLM version is compatible with a constraint.

        Args:
            constraint: Version constraint (e.g., ">=0.6.0").
                       If None, returns True (no constraint).

        Returns:
            Tuple of (is_compatible, message)
        """
        if constraint is None:
            return True, "No version constraint specified"

        if self.vllm_version is None:
            return False, "vLLM is not installed"

        return check_version_constraint(self.vllm_version, constraint)

    def check_plugin_compatibility(
        self,
        plugin_name: str,
        vllm_constraint: Optional[str],
    ) -> Tuple[bool, str]:
        """
        Check if a plugin is compatible with the installed vLLM version.

        Args:
            plugin_name: Name of the plugin (for logging)
            vllm_constraint: vLLM version constraint

        Returns:
            Tuple of (is_compatible, message)
        """
        if vllm_constraint is None:
            logger.debug(f"Plugin '{plugin_name}' has no vLLM version constraint")
            return True, "No version constraint"

        is_compatible, message = self.is_compatible(vllm_constraint)

        if is_compatible:
            logger.info(
                f"Plugin '{plugin_name}' is compatible with vLLM {self.vllm_version} "
                f"(requires {vllm_constraint})"
            )
        else:
            logger.warning(
                f"Plugin '{plugin_name}' is NOT compatible with vLLM {self.vllm_version} "
                f"(requires {vllm_constraint})"
            )

        return is_compatible, message
