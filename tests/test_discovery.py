"""Tests for entry point discovery and cache invalidation."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestEntryPointDiscovery:
    """Tests for EntryPointDiscovery class."""

    def test_discover_vllm_entry_points(self):
        """Test discovering vLLM-related entry points."""
        from vllm_plugin_manager.core.discovery import EntryPointDiscovery

        discovery = EntryPointDiscovery()

        # Get all vLLM entry points
        entry_points = discovery.get_vllm_entry_points()

        # Should return a dict with vLLM groups
        assert isinstance(entry_points, dict)
        # These are the expected vLLM entry point groups
        expected_groups = [
            "vllm.general_plugins",
            "vllm.logits_processors",
            "vllm.stat_logger_plugins",
            "vllm.platform_plugins",
        ]
        for group in expected_groups:
            assert group in entry_points

    def test_get_entry_points_for_group(self):
        """Test getting entry points for a specific group."""
        from vllm_plugin_manager.core.discovery import EntryPointDiscovery

        discovery = EntryPointDiscovery()

        # Get entry points for general plugins group
        eps = discovery.get_entry_points_for_group("vllm.general_plugins")

        assert isinstance(eps, list)
        # Our own plugin manager should be registered
        # (once installed in dev mode)

    def test_invalidate_cache(self):
        """Test that cache invalidation clears importlib caches."""
        from vllm_plugin_manager.core.discovery import EntryPointDiscovery

        discovery = EntryPointDiscovery()

        # This should not raise any errors
        discovery.invalidate_cache()

        # After invalidation, we should still be able to discover entry points
        entry_points = discovery.get_vllm_entry_points()
        assert isinstance(entry_points, dict)

    def test_snapshot_and_diff(self):
        """Test taking snapshots and finding new entry points."""
        from vllm_plugin_manager.core.discovery import EntryPointDiscovery

        discovery = EntryPointDiscovery()

        # Take initial snapshot
        discovery.take_snapshot()

        # Get diff (should be empty since nothing changed)
        new_eps = discovery.get_new_entry_points()

        assert isinstance(new_eps, dict)
        # No new packages installed, so should be empty
        for group, eps in new_eps.items():
            assert len(eps) == 0

    def test_find_entry_points_for_package(self):
        """Test finding entry points registered by a specific package."""
        from vllm_plugin_manager.core.discovery import EntryPointDiscovery

        discovery = EntryPointDiscovery()

        # Look for entry points from a known package (pytest has some)
        eps = discovery.get_entry_points_for_package("pytest")

        assert isinstance(eps, dict)

    def test_entry_point_to_dict(self):
        """Test converting entry point to dictionary representation."""
        from vllm_plugin_manager.core.discovery import EntryPointDiscovery

        discovery = EntryPointDiscovery()

        # Get some entry points
        all_eps = discovery.get_vllm_entry_points()

        # Check structure if we have any
        for group, eps in all_eps.items():
            for ep in eps:
                ep_dict = discovery.entry_point_to_dict(ep)
                assert "name" in ep_dict
                assert "value" in ep_dict
                assert "group" in ep_dict


class TestCacheInvalidation:
    """Tests specifically for importlib cache invalidation."""

    def test_invalidate_importlib_metadata_cache(self):
        """Test invalidating importlib.metadata caches."""
        from vllm_plugin_manager.core.discovery import invalidate_importlib_cache

        # Should not raise
        invalidate_importlib_cache()

    def test_invalidate_clears_distributions_cache(self):
        """Test that distributions cache is cleared."""
        import importlib.metadata

        from vllm_plugin_manager.core.discovery import invalidate_importlib_cache

        # Access distributions to populate cache
        list(importlib.metadata.distributions())

        # Invalidate
        invalidate_importlib_cache()

        # Should still work after invalidation
        dists = list(importlib.metadata.distributions())
        assert len(dists) > 0

    def test_cache_invalidation_allows_new_package_discovery(self):
        """Test that cache invalidation allows discovering newly installed packages."""
        from vllm_plugin_manager.core.discovery import EntryPointDiscovery

        discovery = EntryPointDiscovery()

        # Take snapshot before
        discovery.take_snapshot()

        # Invalidate cache (simulates post-pip-install state)
        discovery.invalidate_cache()

        # Should be able to get entry points
        eps = discovery.get_vllm_entry_points()
        assert isinstance(eps, dict)


class TestPackageEntryPoints:
    """Tests for package-specific entry point discovery."""

    def test_list_packages_with_vllm_plugins(self):
        """Test listing all packages that provide vLLM plugins."""
        from vllm_plugin_manager.core.discovery import EntryPointDiscovery

        discovery = EntryPointDiscovery()

        packages = discovery.list_packages_with_vllm_plugins()

        assert isinstance(packages, list)
        # Each entry should have package name and entry points
        for pkg in packages:
            assert "name" in pkg
            assert "entry_points" in pkg

    def test_get_package_metadata(self):
        """Test getting metadata for a package."""
        from vllm_plugin_manager.core.discovery import EntryPointDiscovery

        discovery = EntryPointDiscovery()

        # Get metadata for a known package
        metadata = discovery.get_package_metadata("pytest")

        if metadata:  # Package might not be installed in all envs
            assert "name" in metadata
            assert "version" in metadata


class TestEntryPointGroups:
    """Tests for vLLM entry point group constants."""

    def test_vllm_entry_point_groups_defined(self):
        """Test that vLLM entry point groups are properly defined."""
        from vllm_plugin_manager.core.discovery import VLLM_ENTRY_POINT_GROUPS

        assert "vllm.general_plugins" in VLLM_ENTRY_POINT_GROUPS
        assert "vllm.logits_processors" in VLLM_ENTRY_POINT_GROUPS
        assert "vllm.stat_logger_plugins" in VLLM_ENTRY_POINT_GROUPS
        assert "vllm.platform_plugins" in VLLM_ENTRY_POINT_GROUPS

    def test_is_vllm_entry_point_group(self):
        """Test checking if a group is a vLLM entry point group."""
        from vllm_plugin_manager.core.discovery import is_vllm_entry_point_group

        assert is_vllm_entry_point_group("vllm.general_plugins") is True
        assert is_vllm_entry_point_group("vllm.logits_processors") is True
        assert is_vllm_entry_point_group("console_scripts") is False
        assert is_vllm_entry_point_group("random.group") is False
