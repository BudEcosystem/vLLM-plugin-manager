"""Source handlers for installing plugins."""

from .installer import PackageInstaller, InstallerError

__all__ = ["PackageInstaller", "InstallerError"]
