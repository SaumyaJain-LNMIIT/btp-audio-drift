"""Reproducibility snapshot.

Every run writes a small JSON manifest next to its outputs recording the exact
setup that produced the results: resolved config, seed(s), library versions,
Python/platform, a content hash of the config, and the git commit if the code
lives in a repo. This lets any number in the report be traced back to the code
and environment that generated it — a standard requirement for a result to be
considered reproducible.

Usage:
    from utils.reproducibility import write_run_manifest
    write_run_manifest(output_dir, config=cfg, seeds=[seed])
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

# Libraries whose versions materially affect results and are worth pinning.
_TRACKED_PACKAGES = [
    "numpy",
    "scipy",
    "scikit-learn",
    "torch",
    "river",
    "librosa",
    "soundata",
    "soundfile",
    "PyYAML",
    "matplotlib",
    "tqdm",
]


def _pkg_version(name: str) -> str | None:
    """Installed version of a package, or None if it isn't installed."""
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _git_commit() -> str | None:
    """Current git commit (with a '-dirty' suffix if there are uncommitted
    changes), or None if this isn't a git checkout / git isn't available."""
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=3,
        )
        if head.returncode != 0:
            return None
        commit = head.stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=3,
        )
        if status.returncode == 0 and status.stdout.strip():
            commit += "-dirty"
        return commit
    except (OSError, subprocess.SubprocessError):
        return None


def _config_hash(config: Any) -> str | None:
    """Stable SHA-256 (first 12 hex chars) of the config, so two runs can be
    confirmed to have used byte-identical settings."""
    if config is None:
        return None
    try:
        blob = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:12]
    except (TypeError, ValueError):
        return None


def capture_environment(
    config: Any = None,
    seeds: list[int] | None = None,
    extra: dict | None = None,
) -> dict:
    """Build the reproducibility snapshot as a plain dict."""
    snap: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "seeds": list(seeds) if seeds is not None else None,
        "config_hash": _config_hash(config),
        "packages": {name: _pkg_version(name) for name in _TRACKED_PACKAGES},
    }
    if config is not None:
        snap["config"] = config
    if extra:
        snap.update(extra)
    return snap


def write_run_manifest(
    output_dir: str | Path,
    config: Any = None,
    seeds: list[int] | None = None,
    extra: dict | None = None,
    filename: str = "run_manifest.json",
) -> tuple[Path, dict]:
    """Write the snapshot to ``output_dir/filename`` and return (path, snapshot)."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    snap = capture_environment(config=config, seeds=seeds, extra=extra)
    path = out / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(snap, f, indent=2, default=str)
    return path, snap