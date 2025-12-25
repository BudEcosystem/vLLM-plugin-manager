"""Entry point discovery and cache invalidation for vLLM plugins."""

import importlib.metadata
import logging
import sys
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# vLLM entry point groups
VLLM_ENTRY_POINT_GROUPS = [
    "vllm.general_plugins",
    "vllm.logits_processors",
    "vllm.stat_logger_plugins",
    "vllm.platform_plugins",
]


def is_vllm_entry_point_group(group: str) -> bool:
    """Check if a group name is a vLLM entry point group."""
    return group in VLLM_ENTRY_POINT_GROUPS


def invalidate_importlib_cache() -> None:
    """
    Invalidate importlib.metadata caches to discover newly installed packages.

    This is critical for discovering plugins installed via pip during runtime.
    Without this, Python's entry_points() won't see new packages until restart.
    """
    # Clear the distributions cache
    # Different approaches for different Python versions
    if hasattr(importlib.metadata, "_adapters"):
        # Python 3.10+
        if hasattr(importlib.metadata._adapters, "_entries"):
            try:
                importlib.metadata._adapters._entries.cache_clear()
            except Exception:
                pass

    # Clear sys.meta_path finder caches
    for finder in sys.meta_path:
        if hasattr(finder, "invalidate_caches"):
            try:
                finder.invalidate_caches()
            except Exception:
                pass

    # Invalidate path importer caches
    importlib.invalidate_caches()

    # Clear any cached distributions
    # This forces re-discovery on next entry_points() call
    if hasattr(importlib.metadata, "distributions"):
        # Force re-iteration of distributions
        try:
            # Accessing distributions triggers cache refresh
            list(importlib.metadata.distributions())
        except Exception:
            pass

    logger.debug("Invalidated importlib caches")


class EntryPointDiscovery:
    """
    Discovers and tracks entry points for vLLM plugins.

    Supports taking snapshots to detect newly installed plugins.
    """

    def __init__(self):
        """Initialize the discovery system."""
        self._snapshot: Optional[Dict[str, List[Dict[str, str]]]] = None

    def invalidate_cache(self) -> None:
        """Invalidate importlib caches."""
        invalidate_importlib_cache()

    def get_vllm_entry_points(self) -> Dict[str, List[Any]]:
        """
        Get all vLLM-related entry points.

        Returns:
            Dict mapping group name to list of entry points
        """
        result = {group: [] for group in VLLM_ENTRY_POINT_GROUPS}

        try:
            # Python 3.10+ API
            eps = importlib.metadata.entry_points()

            for group in VLLM_ENTRY_POINT_GROUPS:
                if hasattr(eps, "select"):
                    # Python 3.10+
                    result[group] = list(eps.select(group=group))
                elif hasattr(eps, "get"):
                    # Python 3.9
                    result[group] = list(eps.get(group, []))
                elif isinstance(eps, dict):
                    # Fallback for older API
                    result[group] = list(eps.get(group, []))

        except Exception as e:
            logger.error(f"Error discovering entry points: {e}")

        return result

    def get_entry_points_for_group(self, group: str) -> List[Any]:
        """Get entry points for a specific group."""
        try:
            eps = importlib.metadata.entry_points()

            if hasattr(eps, "select"):
                return list(eps.select(group=group))
            elif hasattr(eps, "get"):
                return list(eps.get(group, []))
            elif isinstance(eps, dict):
                return list(eps.get(group, []))

        except Exception as e:
            logger.error(f"Error getting entry points for {group}: {e}")

        return []

    def take_snapshot(self) -> None:
        """Take a snapshot of current entry points for later comparison."""
        self._snapshot = {}

        for group, eps in self.get_vllm_entry_points().items():
            self._snapshot[group] = [self.entry_point_to_dict(ep) for ep in eps]

        logger.debug(f"Took snapshot with {sum(len(v) for v in self._snapshot.values())} entry points")

    def get_new_entry_points(self) -> Dict[str, List[Any]]:
        """
        Get entry points added since the last snapshot.

        Returns:
            Dict mapping group name to list of new entry points
        """
        if self._snapshot is None:
            logger.warning("No snapshot taken, returning empty diff")
            return {group: [] for group in VLLM_ENTRY_POINT_GROUPS}

        # Get current entry points
        current = self.get_vllm_entry_points()
        new_eps = {}

        for group in VLLM_ENTRY_POINT_GROUPS:
            current_eps = current.get(group, [])
            snapshot_eps = self._snapshot.get(group, [])

            # Convert snapshot to set of (name, value) tuples for comparison
            snapshot_set = {(ep["name"], ep["value"]) for ep in snapshot_eps}

            # Find new ones
            new_eps[group] = [
                ep
                for ep in current_eps
                if (ep.name, ep.value) not in snapshot_set
            ]

        return new_eps

    def get_entry_points_for_package(self, package_name: str) -> Dict[str, List[Any]]:
        """
        Get all entry points registered by a specific package.

        Args:
            package_name: Name of the package to look up

        Returns:
            Dict mapping group name to list of entry points from that package
        """
        result = {}

        try:
            # Get the distribution for this package
            dist = importlib.metadata.distribution(package_name)

            # Get all entry points from this distribution
            if hasattr(dist, "entry_points"):
                eps = dist.entry_points
                for ep in eps:
                    group = ep.group
                    if group not in result:
                        result[group] = []
                    result[group].append(ep)

        except importlib.metadata.PackageNotFoundError:
            logger.debug(f"Package '{package_name}' not found")
        except Exception as e:
            logger.error(f"Error getting entry points for package {package_name}: {e}")

        return result

    def list_packages_with_vllm_plugins(self) -> List[Dict[str, Any]]:
        """
        List all packages that provide vLLM plugins.

        Returns:
            List of dicts with package name and entry points
        """
        packages = {}

        for group, eps in self.get_vllm_entry_points().items():
            for ep in eps:
                # Get package name from entry point
                try:
                    if hasattr(ep, "dist") and ep.dist:
                        pkg_name = ep.dist.name
                    else:
                        # Fallback: try to extract from entry point
                        pkg_name = getattr(ep, "_for", None) or "unknown"

                    if pkg_name not in packages:
                        packages[pkg_name] = {
                            "name": pkg_name,
                            "entry_points": [],
                        }

                    packages[pkg_name]["entry_points"].append({
                        "group": group,
                        "name": ep.name,
                        "value": ep.value,
                    })

                except Exception:
                    continue

        return list(packages.values())

    def get_package_metadata(self, package_name: str) -> Optional[Dict[str, str]]:
        """
        Get metadata for a package.

        Args:
            package_name: Name of the package

        Returns:
            Dict with name, version, and other metadata, or None if not found
        """
        try:
            dist = importlib.metadata.distribution(package_name)
            return {
                "name": dist.metadata.get("Name", package_name),
                "version": dist.metadata.get("Version", "unknown"),
                "summary": dist.metadata.get("Summary", ""),
                "author": dist.metadata.get("Author", ""),
            }
        except importlib.metadata.PackageNotFoundError:
            return None
        except Exception as e:
            logger.error(f"Error getting metadata for {package_name}: {e}")
            return None

    @staticmethod
    def entry_point_to_dict(ep: Any) -> Dict[str, str]:
        """
        Convert an entry point to a dictionary representation.

        Args:
            ep: Entry point object

        Returns:
            Dict with name, value, and group
        """
        return {
            "name": ep.name,
            "value": ep.value,
            "group": getattr(ep, "group", ""),
        }
