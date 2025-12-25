"""Package installer for vLLM plugins."""

import importlib.metadata
import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import PluginSpec

logger = logging.getLogger(__name__)


class InstallerError(Exception):
    """Raised when plugin installation fails."""

    pass


class PackageInstaller:
    """
    Installs Python packages from various sources using pip.

    Supports:
    - PyPI packages
    - Git repositories
    - Local directories
    """

    DEFAULT_TIMEOUT = 300  # 5 minutes

    def __init__(self, timeout: Optional[float] = None):
        """
        Initialize the installer.

        Args:
            timeout: Timeout for pip commands in seconds
        """
        self.timeout = timeout or self.DEFAULT_TIMEOUT

    def _run_pip(self, args: list) -> Tuple[bool, str]:
        """
        Run a pip command.

        Args:
            args: Arguments to pass to pip

        Returns:
            Tuple of (success, output)
        """
        cmd = [sys.executable, "-m", "pip"] + args

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )

            output = result.stdout + result.stderr
            success = result.returncode == 0

            if not success:
                logger.warning(f"pip command failed: {' '.join(cmd)}")
                logger.warning(f"Output: {output}")

            return success, output

        except subprocess.TimeoutExpired:
            return False, f"Command timed out after {self.timeout} seconds"
        except Exception as e:
            return False, f"Error running pip: {e}"

    def install_pypi(
        self,
        package: str,
        version: Optional[str] = None,
        upgrade: bool = False,
    ) -> Tuple[bool, str]:
        """
        Install a package from PyPI.

        Args:
            package: Package name
            version: Version specifier (e.g., ">=1.0.0")
            upgrade: Whether to upgrade if already installed

        Returns:
            Tuple of (success, message)
        """
        # Build package spec
        if version:
            package_spec = f"{package}{version}"
        else:
            package_spec = package

        args = ["install"]
        if upgrade:
            args.append("--upgrade")
        args.append(package_spec)

        logger.info(f"Installing from PyPI: {package_spec}")
        return self._run_pip(args)

    def install_git(
        self,
        url: str,
        ref: Optional[str] = None,
        subdirectory: Optional[str] = None,
        editable: bool = False,
    ) -> Tuple[bool, str]:
        """
        Install a package from a Git repository.

        Args:
            url: Git repository URL
            ref: Branch, tag, or commit to install
            subdirectory: Subdirectory within repo containing the package
            editable: Install in editable mode

        Returns:
            Tuple of (success, message)
        """
        # Build git URL
        git_url = f"git+{url}"
        if ref:
            git_url += f"@{ref}"
        if subdirectory:
            git_url += f"#subdirectory={subdirectory}"

        args = ["install"]
        if editable:
            args.append("-e")
        args.append(git_url)

        logger.info(f"Installing from Git: {git_url}")
        return self._run_pip(args)

    def install_local(
        self,
        path: Path,
        editable: bool = True,
    ) -> Tuple[bool, str]:
        """
        Install a package from a local directory.

        Args:
            path: Path to the package directory
            editable: Install in editable mode (default True for local)

        Returns:
            Tuple of (success, message)
        """
        args = ["install"]
        if editable:
            args.append("-e")
        args.append(str(path))

        logger.info(f"Installing from local: {path}")
        return self._run_pip(args)

    def install_from_spec(self, spec: "PluginSpec") -> Tuple[bool, str]:
        """
        Install a package from a PluginSpec.

        Args:
            spec: Plugin specification

        Returns:
            Tuple of (success, message)

        Raises:
            InstallerError: If source type is unknown
        """
        if spec.source == "pypi":
            return self.install_pypi(
                package=spec.package or spec.name,
                version=spec.version,
            )

        elif spec.source == "git":
            if not spec.url:
                raise InstallerError(f"Git plugin '{spec.name}' missing URL")
            return self.install_git(
                url=spec.url,
                ref=spec.ref,
                subdirectory=spec.subdirectory,
            )

        elif spec.source == "local":
            if not spec.path:
                raise InstallerError(f"Local plugin '{spec.name}' missing path")
            return self.install_local(
                path=Path(spec.path),
                editable=spec.editable,
            )

        else:
            raise InstallerError(f"Unknown source type: {spec.source}")

    def uninstall(self, package: str) -> Tuple[bool, str]:
        """
        Uninstall a package.

        Args:
            package: Package name to uninstall

        Returns:
            Tuple of (success, message)
        """
        args = ["uninstall", "-y", package]

        logger.info(f"Uninstalling: {package}")
        return self._run_pip(args)

    def is_installed(self, package: str) -> bool:
        """
        Check if a package is installed.

        Args:
            package: Package name to check

        Returns:
            True if installed, False otherwise
        """
        try:
            importlib.metadata.distribution(package)
            return True
        except importlib.metadata.PackageNotFoundError:
            return False

    def get_installed_version(self, package: str) -> Optional[str]:
        """
        Get the installed version of a package.

        Args:
            package: Package name

        Returns:
            Version string or None if not installed
        """
        try:
            dist = importlib.metadata.distribution(package)
            return dist.metadata.get("Version")
        except importlib.metadata.PackageNotFoundError:
            return None
