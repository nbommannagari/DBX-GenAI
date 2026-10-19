# _lib/library_installer.py
#
# Config-driven library installation for GenAI courses.
# Reads library groups from a config YAML, resolves packages + versions
# from the course's dependencies.toml, and installs them via pip.

import subprocess
import sys
import tomllib
import yaml
from pathlib import Path


def _build_spec(pkg: str, ver: str) -> str:
    """Build a pip install spec from a package name and version string."""
    if ver == "*":
        return pkg
    if ver[0] in ">=<~!":
        return f"{pkg}{ver}"
    return f"{pkg}=={ver}"


def _find_deps_file(config_path: str | Path) -> Path:
    """Locate the dependencies.toml for a given config file.

    Walks up from the config's directory looking for dependencies.toml
    in the course includes folder (the config's grandparent directory).
    """
    config_dir = Path(config_path).resolve().parent
    # config is at <course-includes>/config/config_N.yaml
    # dependencies.toml is at <course-includes>/dependencies.toml
    course_dir = config_dir.parent
    deps_file = course_dir / "dependencies.toml"
    if deps_file.exists():
        return deps_file
    raise FileNotFoundError(
        f"dependencies.toml not found at {deps_file}. "
        f"Each course includes folder must have its own dependencies.toml."
    )


def resolve_packages(config_path: str | Path) -> list[str]:
    """Resolve library groups from a config YAML into a deduplicated package list.

    Parameters
    ----------
    config_path : str | Path
        Path to a course config YAML containing a ``libraries.groups`` list.

    Returns
    -------
    list[str]
        Pip install specs, e.g. ``["pyyaml==6.0.3", "backoff==2.2.1"]``.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}

    groups = (cfg.get("libraries") or {}).get("groups", [])
    if not groups:
        return []

    deps_file = _find_deps_file(config_path)
    with open(deps_file, "rb") as f:
        deps = tomllib.load(f)

    missing = [g for g in groups if g not in deps]
    if missing:
        raise KeyError(
            f"Library group(s) {missing} not found in {deps_file}. "
            f"Available: {sorted(deps.keys())}"
        )

    # Collect packages; last group wins on version conflicts.
    seen: dict[str, str] = {}
    for group in groups:
        for pkg, ver in deps[group].items():
            seen[pkg] = _build_spec(pkg, ver)

    return list(seen.values())


def _ensure_uv() -> None:
    """Make the ``uv`` installer available in the current interpreter.

    ``uv`` resolves the full pinned package set together (see
    :func:`install_libraries`); it may not be present in every course's base
    environment, so bootstrap it with pip if needed. This bootstrap install of
    the tool is unpinned; a course that pins ``uv`` in its groups still gets
    that pinned version installed as part of the batch below.
    """
    check = subprocess.run(
        [sys.executable, "-m", "uv", "--version"],
        capture_output=True, text=True,
    )
    if check.returncode == 0:
        return
    boot = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "uv"],
        capture_output=True, text=True,
    )
    if boot.returncode != 0:
        print("\n  Failed to bootstrap uv:")
        print((boot.stdout or "").rstrip())
        print((boot.stderr or "").rstrip())
        raise subprocess.CalledProcessError(boot.returncode, "pip install uv")


def install_libraries(config_path: str | Path) -> None:
    """Install libraries declared in a config YAML's ``libraries.groups`` section.

    Reads group names from the config, resolves them against the course's
    ``dependencies.toml``, and installs the resulting package list with
    ``uv pip install`` in a single command.

    Installing all pinned specs at once (rather than one at a time with
    ``pip -U``) lets the resolver honor every pin together. pip's backtracking
    resolver hit ``ResolutionTooDeep`` on some sets (e.g. databricks-langchain's
    openai/openai-agents tree); uv resolves the same set in seconds. The full
    ``uv`` output is surfaced on failure instead of being masked.

    Parameters
    ----------
    config_path : str | Path
        Path to a course config YAML.
    """
    packages = resolve_packages(config_path)
    if not packages:
        print("No library groups specified in config. Skipping install.")
        return

    print(f"Installing {len(packages)} package(s) with uv...")
    for p in packages:
        print(f"  {p}")

    _ensure_uv()

    # Single batched install so the resolver sees all pins together.
    result = subprocess.run(
        [sys.executable, "-m", "uv", "pip", "install", "--python", sys.executable, *packages],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"\n  uv pip install failed (exit {result.returncode})")
        if result.stdout:
            print("  --- uv stdout ---")
            print(result.stdout.rstrip())
        if result.stderr:
            print("  --- uv stderr ---")
            print(result.stderr.rstrip())
        raise subprocess.CalledProcessError(
            result.returncode, "uv pip install",
            output=result.stdout, stderr=result.stderr,
        )

    # Report any broken-but-installed state (conflicts that don't fail the
    # install itself) instead of letting it surface as an import error later.
    check = subprocess.run(
        [sys.executable, "-m", "uv", "pip", "check"],
        capture_output=True, text=True,
    )
    if check.returncode != 0:
        print("\nWARNING: uv pip check reported dependency conflicts:")
        print((check.stdout or check.stderr).rstrip())

    print("Library installation complete.")
