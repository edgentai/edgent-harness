"""Scheduler (§8.1): run ready nodes in parallel, checkpoint after each completion."""
from __future__ import annotations

import concurrent.futures
import threading
from typing import Any, Callable, Optional

from ..core.types import NodeStatus
from .dag import Node, StateDB, TaskDAG

WorkerFn = Callable[[Node], tuple[bool, dict[str, Any], dict[str, Any], Optional[str]]]


class Scheduler:
    def __init__(self, db: StateDB, dag: TaskDAG, max_parallel: int = 3, tracer: Any = None, on_dispatch: Optional[Callable[[Node], None]] = None) -> None:
        self.db = db
        self.dag = dag
        self.max_parallel = max(1, max_parallel)
        self.tracer = tracer
        self.on_dispatch = on_dispatch  # test hook (crash injection)
        self._lock = threading.Lock()

    def step(self, worker: WorkerFn, stop: Callable[[], bool]) -> list[Node]:
        """Dispatch every ready node (bounded by max_parallel), wait for the wave, commit each result. Returns finished nodes."""
        ready = self.dag.ready_nodes()[: self.max_parallel]
        if not ready:
            return []
        finished: list[Node] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(ready), thread_name_prefix="worker") as pool:
            futures = {}
            for n in ready:
                if stop():
                    break
                self.dag.start(n.id, owner=f"worker-{threading.get_ident()}")
                if self.tracer:
                    self.tracer.record("node", {"id": n.id, "status": "running", "attempt": n.attempts + 1, "role": n.role}, node_id=n.id)
                if self.on_dispatch:
                    self.on_dispatch(n)
                futures[pool.submit(worker, n)] = n
            for fut in concurrent.futures.as_completed(futures):
                n = futures[fut]
                try:
                    ok, output, cost, error = fut.result()
                except Exception as e:  # noqa: BLE001
                    ok, output, cost, error = False, {}, {}, f"worker crashed: {type(e).__name__}: {e}"
                with self._lock:
                    if ok:
                        node = self.dag.complete(n.id, output, cost)
                    else:
                        node = self.dag.fail(n.id, error or "failed", output, cost)
                    self.db.checkpoint(self.dag.run_id, {"node": n.id, "status": node.status.value, "summary": self.dag.summary()})
                if self.tracer:
                    self.tracer.record("node", {"id": n.id, "status": node.status.value, "error": error, "cost": cost}, node_id=n.id)
                finished.append(node)
        return finished
