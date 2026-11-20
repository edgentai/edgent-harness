"""Periodic background heartbeat scheduler (OpenClaw-style proactive wakeups).

Periodically wakes an ``AgentLoop`` session with a configured prompt under the
same permission policy, hooks, and budget ceilings, logging each run cleanly to
``<state_dir>/heartbeat.jsonl``.
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from ..core.utils import ensure_dir, now_iso, truncate


@dataclass
class HeartbeatJob:
    job_id: str
    session_id: str
    prompt: str
    interval_s: float = 300.0
    enabled: bool = True
    max_runs: Optional[int] = None
    stage: Optional[str] = None
    runs: int = 0
    last_run_at: Optional[float] = None
    last_status: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "session_id": self.session_id,
            "prompt": self.prompt,
            "interval_s": self.interval_s,
            "enabled": self.enabled,
            "max_runs": self.max_runs,
            "stage": self.stage,
            "runs": self.runs,
            "last_status": self.last_status,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "HeartbeatJob":
        return cls(
            job_id=str(d["job_id"]),
            session_id=str(d.get("session_id") or d["job_id"]),
            prompt=str(d["prompt"]),
            interval_s=max(0.05, float(d.get("interval_s", 300.0))),
            enabled=bool(d.get("enabled", True)),
            max_runs=int(d["max_runs"]) if d.get("max_runs") is not None else None,
            stage=d.get("stage"),
        )


class HeartbeatScheduler:
    """Daemon thread scheduler that wakes an ``AgentLoop`` on configured intervals."""

    def __init__(
        self,
        runner: Callable[[str, str, Optional[str]], Any],
        *,
        log_path: Optional[str] = None,
        poll_s: float = 0.25,
    ) -> None:
        self._runner = runner
        self.log_path = log_path
        if log_path:
            ensure_dir(os.path.dirname(os.path.abspath(log_path)))
        self.poll_s = max(0.02, poll_s)
        self._jobs: dict[str, HeartbeatJob] = {}
        self._lock = threading.Lock()
        self._session_locks: dict[str, threading.Lock] = {}
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def add_job(self, job: HeartbeatJob) -> None:
        with self._lock:
            self._jobs[job.job_id] = job

    def remove_job(self, job_id: str) -> bool:
        with self._lock:
            return self._jobs.pop(job_id, None) is not None

    def list_jobs(self) -> list[dict[str, Any]]:
        with self._lock:
            return [j.to_dict() for j in self._jobs.values()]

    def load_jobs_file(self, path: str) -> int:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        items = raw.get("jobs", raw) if isinstance(raw, dict) else raw
        added = 0
        for item in items:
            if isinstance(item, dict) and item.get("job_id") and item.get("prompt"):
                self.add_job(HeartbeatJob.from_dict(item))
                added += 1
        return added

    def _get_session_lock(self, session_id: str) -> threading.Lock:
        with self._lock:
            lk = self._session_locks.get(session_id)
            if lk is None:
                lk = threading.Lock()
                self._session_locks[session_id] = lk
            return lk

    def _log(self, rec: dict[str, Any]) -> None:
        if not self.log_path:
            return
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    def trigger_now(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(f"unknown heartbeat job '{job_id}'")
        return self._execute(job)

    def _execute(self, job: HeartbeatJob) -> dict[str, Any]:
        sess_lock = self._get_session_lock(job.session_id)
        with sess_lock:
            t0 = time.monotonic()
            try:
                res = self._runner(job.session_id, job.prompt, job.stage)
                status = getattr(res, "status", "completed")
                reply = getattr(res, "reply", str(res))
                turns = getattr(res, "turns_used", 0)
                tools = getattr(res, "tool_calls_used", 0)
                err = getattr(res, "error", None)
            except Exception as e:  # noqa: BLE001
                status = "error"
                reply = ""
                turns = 0
                tools = 0
                err = f"{type(e).__name__}: {e}"
            with self._lock:
                job.runs += 1
                job.last_run_at = time.monotonic()
                job.last_status = status
                if job.max_runs is not None and job.runs >= job.max_runs:
                    job.enabled = False
            rec = {
                "ts": now_iso(),
                "job_id": job.job_id,
                "session_id": job.session_id,
                "run_number": job.runs,
                "status": status,
                "turns_used": turns,
                "tool_calls_used": tools,
                "duration_s": round(time.monotonic() - t0, 3),
                "reply_preview": truncate(str(reply), 300),
                "error": err,
            }
            self._log(rec)
            return rec

    def _loop(self) -> None:
        while not self._stop.is_set():
            now = time.monotonic()
            due: list[HeartbeatJob] = []
            with self._lock:
                for job in self._jobs.values():
                    if not job.enabled:
                        continue
                    if job.max_runs is not None and job.runs >= job.max_runs:
                        job.enabled = False
                        continue
                    if job.last_run_at is None or (now - job.last_run_at) >= job.interval_s:
                        due.append(job)
            for job in due:
                if self._stop.is_set():
                    break
                self._execute(job)
            self._stop.wait(self.poll_s)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="edgent-heartbeat", daemon=True)
        self._thread.start()

    def stop(self, timeout_s: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout_s)
            self._thread = None
