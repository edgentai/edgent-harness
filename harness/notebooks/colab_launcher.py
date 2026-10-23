"""Colab launcher (§4.1). The notebook does nothing but run this file.

Usage inside Colab:
    !git clone <repo> && cd edgent-harness && python -m harness.notebooks.colab_launcher --model-path Qwen/Qwen3.8-27B

It mounts Drive (when running in Colab), caches weights and state there,
starts the serving engine under a supervisor, runs the health checks, and
resumes any run that was in flight when the previous session died.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request

from ..core.config import HarnessConfig
from ..orchestration.master import Harness
from ..serving.health import HealthChecker
from ..serving.launch import Supervisor, sglang_command, vllm_command
from ..serving.manifest import write_manifest


def mount_drive() -> str | None:
    try:
        from google.colab import drive  # type: ignore

        drive.mount("/content/drive")
        return "/content/drive/MyDrive/edgent-harness"
    except Exception:
        return None


def wait_for(url: str, timeout_s: float = 1800.0) -> bool:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(5)
    return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--engine", choices=["vllm", "sglang"], default="vllm")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--state-dir", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--no-serve", action="store_true", help="assume a server is already running")
    args = ap.parse_args(argv)

    persistent = mount_drive()
    cfg = HarnessConfig.load(args.config)
    cfg.state_dir = args.state_dir or (os.path.join(persistent, "state") if persistent else cfg.state_dir)
    cfg.serving.base_url = f"http://127.0.0.1:{args.port}/v1"
    os.makedirs(cfg.state_dir, exist_ok=True)
    if persistent:
        os.environ.setdefault("HF_HOME", os.path.join(persistent, "hf_cache"))  # weights cached on Drive: cold start costs minutes
    manifest = write_manifest(cfg, os.path.join(cfg.state_dir, "environment_manifest.json"))
    print("environment:", json.dumps(manifest, indent=1)[:800])

    harness = Harness(cfg, backend="openai")
    sup = None
    if not args.no_serve:
        cmd = (vllm_command if args.engine == "vllm" else sglang_command)(cfg.serving, args.model_path, args.port)
        print("launching:", " ".join(cmd))
        sup = Supervisor(cmd, lambda: harness.client.health().get("ok", False), on_restart=harness.resume_all, check_interval_s=30)
        sup.start()
        if not wait_for(f"http://127.0.0.1:{args.port}/v1/models"):
            print("model server did not come up", file=sys.stderr)
            return 1
    hc = HealthChecker(harness.client, cfg.capabilities.native_image).run_all()
    print("health:", json.dumps(hc, indent=1)[:1500])
    resumed = harness.resume_all()
    print(f"resumed {len(resumed)} in-flight run(s)")
    print("ready. Use `python -m harness run ...` against", cfg.serving.base_url)
    try:
        while sup is not None:
            time.sleep(60)
            if harness.session_draining():
                print("session nearing its limit: draining")
                break
    except KeyboardInterrupt:
        pass
    if sup:
        sup.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
