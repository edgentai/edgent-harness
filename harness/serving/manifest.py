"""Known-good environment manifest: GPU, driver, engine, model revision, quantisation, context length."""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from typing import Any, Optional

from ..core.config import HarnessConfig


def _nvidia_smi() -> Optional[dict[str, Any]]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], capture_output=True, text=True, timeout=10).stdout.strip()
        name, mem, drv = [x.strip() for x in out.split(",")[:3]]
        return {"gpu": name, "memory": mem, "driver": drv}
    except Exception:
        return None


def _pkg_version(name: str) -> Optional[str]:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def environment_manifest(config: HarnessConfig) -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "gpu": _nvidia_smi(),
        "engine": {"vllm": _pkg_version("vllm"), "sglang": _pkg_version("sglang"), "torch": _pkg_version("torch")},
        "model": config.serving.model,
        "quantisation": config.serving.quantisation,
        "max_context_length": config.serving.max_context_length,
        "max_concurrent_sequences": config.serving.max_concurrent_sequences,
        "prefix_caching": config.serving.prefix_caching,
        "tools": {t: shutil.which(t) is not None for t in ("ffmpeg", "ffprobe", "tesseract")},
        "env": {k: v for k, v in os.environ.items() if k.startswith("HARNESS_") and "KEY" not in k},
    }


def write_manifest(config: HarnessConfig, path: str) -> dict[str, Any]:
    m = environment_manifest(config)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2)
    return m
