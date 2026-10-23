"""Token and cost accounting per run and per node."""
from __future__ import annotations

import threading
from typing import Any

from ..core.types import Usage


class CostAccountant:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.total = Usage()
        self.by_node: dict[str, Usage] = {}
        self.by_role: dict[str, Usage] = {}
        self.by_effort: dict[str, Usage] = {}

    def charge(self, usage: Usage, node_id: str = "_run", role: str = "_", effort: str = "_") -> None:
        with self._lock:
            self.total = self.total.add(usage)
            self.by_node[node_id] = self.by_node.get(node_id, Usage()).add(usage)
            self.by_role[role] = self.by_role.get(role, Usage()).add(usage)
            self.by_effort[effort] = self.by_effort.get(effort, Usage()).add(usage)

    def report(self) -> dict[str, Any]:
        with self._lock:
            return {
                "total": self.total.to_dict(),
                "by_node": {k: v.to_dict() for k, v in self.by_node.items()},
                "by_role": {k: v.to_dict() for k, v in self.by_role.items()},
                "by_effort": {k: v.to_dict() for k, v in self.by_effort.items()},
            }
