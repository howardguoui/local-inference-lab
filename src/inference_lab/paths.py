"""Where configs and results live.

An MCP client such as Claude Desktop starts the server from its own working directory,
so paths can't simply be relative. Order: $INFERENCE_LAB_HOME, then the current
directory if it has a configs/ folder, then the repository this package was installed
from (an editable install), then the current directory.
"""

from __future__ import annotations

import os
from pathlib import Path


def lab_home() -> Path:
    env = os.environ.get("INFERENCE_LAB_HOME")
    if env:
        return Path(env).expanduser()
    cwd = Path.cwd()
    if (cwd / "configs").is_dir():
        return cwd
    repo = Path(__file__).resolve().parents[2]
    if (repo / "configs" / "backends.yaml").exists():
        return repo
    return cwd


def results_dir() -> Path:
    return lab_home() / "results"


def backends_file() -> Path:
    return lab_home() / "configs" / "backends.yaml"
