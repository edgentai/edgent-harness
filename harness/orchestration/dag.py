"""Persistent task DAG in SQLite (§11.5, §11.6).

Status transitions are enforced in code. Every change is committed before
the next step starts, so a crash resumes from the stored graph. WAL mode
gives the write-ahead journal that §4.4 asks for.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from ..core.types import NodeStatus, Plan, PlanStep
from ..core.utils import ensure_dir, now_iso

TRANSITIONS: dict[NodeStatus, set[NodeStatus]] = {
    NodeStatus.PENDING: {NodeStatus.READY, NodeStatus.BLOCKED, NodeStatus.SKIPPED, NodeStatus.STALE},
    NodeStatus.READY: {NodeStatus.RUNNING, NodeStatus.BLOCKED, NodeStatus.SKIPPED, NodeStatus.STALE, NodeStatus.PENDING},
    NodeStatus.RUNNING: {NodeStatus.DONE, NodeStatus.FAILED, NodeStatus.READY, NodeStatus.STALE},  # READY = resumed after crash
    NodeStatus.DONE: {NodeStatus.STALE},
    NodeStatus.FAILED: {NodeStatus.READY, NodeStatus.SKIPPED, NodeStatus.STALE, NodeStatus.BLOCKED},
    NodeStatus.BLOCKED: {NodeStatus.READY, NodeStatus.SKIPPED, NodeStatus.STALE, NodeStatus.PENDING},
    NodeStatus.SKIPPED: {NodeStatus.STALE},
    NodeStatus.STALE: {NodeStatus.READY, NodeStatus.PENDING, NodeStatus.SKIPPED},
}


class InvalidTransition(Exception):
    pass


@dataclass
class Node:
    run_id: str
    id: str
    description: str
    role: str = "executor"
    status: NodeStatus = NodeStatus.PENDING
    tools: list[str] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    verifier: Optional[str] = None
    expected_output: str = "text"
    inputs: dict[str, Any] = field(default_factory=dict)
    output: Optional[dict[str, Any]] = None
    owner: Optional[str] = None
    attempts: int = 0
    cost: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    generation: int = 1  # plan version that created it
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["status"] = self.status.value
        return d


class StateDB:
    """Run-level records plus the DAG tables. One file for all runs."""

    def __init__(self, path: str) -> None:
        self.path = path
        ensure_dir(__import__("os").path.dirname(path) or ".")
        self._lock = threading.RLock()
        self._local = threading.local()
        self._all_conns: list[sqlite3.Connection] = []
        with self.conn() as c:
            c.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs(
                    run_id TEXT PRIMARY KEY, task TEXT, status TEXT, effort TEXT, problem_type TEXT, contract TEXT, budget TEXT,
                    plan TEXT, artifacts TEXT, result TEXT, replans INTEGER DEFAULT 0, user_id TEXT, created_at TEXT, updated_at TEXT, meta TEXT
                );
                CREATE TABLE IF NOT EXISTS nodes(
                    run_id TEXT, id TEXT, description TEXT, role TEXT, status TEXT, tools TEXT, artifacts TEXT, verifier TEXT,
                    expected_output TEXT, inputs TEXT, output TEXT, owner TEXT, attempts INTEGER, cost TEXT, error TEXT, generation INTEGER,
                    updated_at TEXT, PRIMARY KEY(run_id, id)
                );
                CREATE TABLE IF NOT EXISTS edges(run_id TEXT, from_id TEXT, to_id TEXT, PRIMARY KEY(run_id, from_id, to_id));
                CREATE TABLE IF NOT EXISTS history(seq INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, node_id TEXT, from_status TEXT, to_status TEXT, ts TEXT, reason TEXT);
                CREATE TABLE IF NOT EXISTS checkpoints(run_id TEXT, seq INTEGER, ts TEXT, snapshot TEXT, PRIMARY KEY(run_id, seq));
                """
            )

    def conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "conn", None)
        if c is None:
            c = sqlite3.connect(self.path, timeout=30, isolation_level=None, check_same_thread=False)
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("PRAGMA synchronous=NORMAL")
            c.execute("PRAGMA busy_timeout=30000")
            c.row_factory = sqlite3.Row
            self._local.conn = c
            with self._lock:
                self._all_conns.append(c)
        return c

    def close(self) -> None:
        with self._lock:
            for c in self._all_conns:
                try:
                    c.close()
                except Exception:
                    pass
            self._all_conns.clear()
        self._local = threading.local()

    # runs
    def create_run(self, run_id: str, task: str, user_id: str = "default", meta: Optional[dict[str, Any]] = None) -> None:
        with self._lock:
            self.conn().execute("INSERT OR REPLACE INTO runs(run_id, task, status, user_id, created_at, updated_at, meta, replans) VALUES (?,?,?,?,?,?,?,0)",
                                (run_id, task, "created", user_id, now_iso(), now_iso(), json.dumps(meta or {})))

    def update_run(self, run_id: str, **fields: Any) -> None:
        if not fields:
            return
        cols = []
        vals: list[Any] = []
        for k, v in fields.items():
            cols.append(f"{k}=?")
            vals.append(json.dumps(v, default=str) if isinstance(v, (dict, list)) else v)
        cols.append("updated_at=?")
        vals.append(now_iso())
        vals.append(run_id)
        with self._lock:
            self.conn().execute(f"UPDATE runs SET {', '.join(cols)} WHERE run_id=?", vals)

    def get_run(self, run_id: str) -> Optional[dict[str, Any]]:
        row = self.conn().execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        for k in ("contract", "budget", "plan", "artifacts", "result", "meta"):
            if d.get(k):
                try:
                    d[k] = json.loads(d[k])
                except Exception:
                    pass
        return d

    def runs(self, status: Optional[str] = None) -> list[dict[str, Any]]:
        q = "SELECT run_id, task, status, effort, created_at, updated_at FROM runs"
        args: list[Any] = []
        if status:
            q += " WHERE status=?"
            args.append(status)
        return [dict(r) for r in self.conn().execute(q + " ORDER BY created_at DESC", args).fetchall()]

    def in_flight(self) -> list[str]:
        """Runs a supervisor should re-attach to after a restart (§4.4)."""
        return [r["run_id"] for r in self.conn().execute("SELECT run_id FROM runs WHERE status IN ('created','planning','running','replanning','synthesising')").fetchall()]

    def checkpoint(self, run_id: str, snapshot: dict[str, Any]) -> int:
        with self._lock:
            row = self.conn().execute("SELECT COALESCE(MAX(seq),0) FROM checkpoints WHERE run_id=?", (run_id,)).fetchone()
            seq = int(row[0]) + 1
            self.conn().execute("INSERT INTO checkpoints VALUES (?,?,?,?)", (run_id, seq, now_iso(), json.dumps(snapshot, default=str)))
        return seq


class TaskDAG:
    def __init__(self, db: StateDB, run_id: str) -> None:
        self.db = db
        self.run_id = run_id

    # -- construction -----------------------------------------------------
    def load_plan(self, plan: Plan, generation: int = 1) -> None:
        with self.db._lock:
            c = self.db.conn()
            for s in plan.steps:
                self._insert(s, generation)
            for s in plan.steps:
                for dep in s.depends_on:
                    c.execute("INSERT OR IGNORE INTO edges VALUES (?,?,?)", (self.run_id, dep, s.id))
            self.refresh_ready()

    def _insert(self, s: PlanStep, generation: int) -> None:
        self.db.conn().execute(
            "INSERT OR REPLACE INTO nodes VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (self.run_id, s.id, s.description, s.role, NodeStatus.PENDING.value, json.dumps(s.tools), json.dumps(s.artifacts), s.verifier,
             s.expected_output, "{}", None, None, 0, "{}", None, generation, now_iso()),
        )
        self._history(s.id, None, NodeStatus.PENDING, "created")

    # -- queries ------------------------------------------------------------
    def node(self, node_id: str) -> Node:
        row = self.db.conn().execute("SELECT * FROM nodes WHERE run_id=? AND id=?", (self.run_id, node_id)).fetchone()
        if not row:
            raise KeyError(node_id)
        return self._row(row)

    def nodes(self, status: Optional[NodeStatus] = None) -> list[Node]:
        q = "SELECT * FROM nodes WHERE run_id=?"
        args: list[Any] = [self.run_id]
        if status:
            q += " AND status=?"
            args.append(status.value)
        return [self._row(r) for r in self.db.conn().execute(q + " ORDER BY rowid", args).fetchall()]

    def _row(self, r: sqlite3.Row) -> Node:
        return Node(r["run_id"], r["id"], r["description"], r["role"], NodeStatus(r["status"]), json.loads(r["tools"] or "[]"), json.loads(r["artifacts"] or "[]"),
                    r["verifier"], r["expected_output"], json.loads(r["inputs"] or "{}"), json.loads(r["output"]) if r["output"] else None, r["owner"], r["attempts"],
                    json.loads(r["cost"] or "{}"), r["error"], r["generation"], r["updated_at"])

    def deps(self, node_id: str) -> list[str]:
        return [r["from_id"] for r in self.db.conn().execute("SELECT from_id FROM edges WHERE run_id=? AND to_id=?", (self.run_id, node_id)).fetchall()]

    def dependents(self, node_id: str) -> list[str]:
        return [r["to_id"] for r in self.db.conn().execute("SELECT to_id FROM edges WHERE run_id=? AND from_id=?", (self.run_id, node_id)).fetchall()]

    def descendants(self, node_id: str) -> list[str]:
        seen: list[str] = []
        stack = [node_id]
        while stack:
            cur = stack.pop()
            for d in self.dependents(cur):
                if d not in seen:
                    seen.append(d)
                    stack.append(d)
        return seen

    def ready_nodes(self) -> list[Node]:
        return self.nodes(NodeStatus.READY)

    def all_done(self) -> bool:
        return all(n.status in (NodeStatus.DONE, NodeStatus.SKIPPED) for n in self.nodes())

    def terminal(self) -> bool:
        """No node can make progress: everything is done/skipped, or remaining ones are failed/blocked with no ready work."""
        statuses = {n.status for n in self.nodes()}
        return not statuses & {NodeStatus.READY, NodeStatus.RUNNING, NodeStatus.PENDING}

    def summary(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for n in self.nodes():
            out[n.status.value] = out.get(n.status.value, 0) + 1
        return out

    # -- transitions --------------------------------------------------------
    def set_status(self, node_id: str, to: NodeStatus, reason: str = "", **fields: Any) -> Node:
        with self.db._lock:
            n = self.node(node_id)
            if to not in TRANSITIONS[n.status] and to != n.status:
                raise InvalidTransition(f"{node_id}: {n.status.value} -> {to.value} not allowed")
            cols = ["status=?", "updated_at=?"]
            vals: list[Any] = [to.value, now_iso()]
            for k, v in fields.items():
                cols.append(f"{k}=?")
                vals.append(json.dumps(v, default=str) if isinstance(v, (dict, list)) else v)
            vals += [self.run_id, node_id]
            self.db.conn().execute(f"UPDATE nodes SET {', '.join(cols)} WHERE run_id=? AND id=?", vals)
            self._history(node_id, n.status, to, reason)
            return self.node(node_id)

    def _history(self, node_id: str, frm: Optional[NodeStatus], to: NodeStatus, reason: str) -> None:
        self.db.conn().execute("INSERT INTO history(run_id,node_id,from_status,to_status,ts,reason) VALUES (?,?,?,?,?,?)",
                               (self.run_id, node_id, frm.value if frm else None, to.value, now_iso(), reason))

    def history(self, node_id: Optional[str] = None) -> list[dict[str, Any]]:
        q = "SELECT node_id, from_status, to_status, ts, reason FROM history WHERE run_id=?"
        args: list[Any] = [self.run_id]
        if node_id:
            q += " AND node_id=?"
            args.append(node_id)
        return [dict(r) for r in self.db.conn().execute(q + " ORDER BY seq", args).fetchall()]

    def refresh_ready(self) -> list[str]:
        """Pending/blocked nodes whose dependencies are all done become ready; nodes with a failed/skipped dep become blocked."""
        changed = []
        with self.db._lock:
            for n in self.nodes():
                if n.status not in (NodeStatus.PENDING, NodeStatus.BLOCKED, NodeStatus.STALE):
                    continue
                dep_status = [self.node(d).status for d in self.deps(n.id)]
                if all(s == NodeStatus.DONE for s in dep_status):
                    self.set_status(n.id, NodeStatus.READY, "dependencies satisfied")
                    changed.append(n.id)
                elif any(s in (NodeStatus.FAILED, NodeStatus.SKIPPED) for s in dep_status) and n.status != NodeStatus.BLOCKED:
                    self.set_status(n.id, NodeStatus.BLOCKED, "dependency failed")
                    changed.append(n.id)
        return changed

    def start(self, node_id: str, owner: str) -> Node:
        n = self.node(node_id)
        return self.set_status(node_id, NodeStatus.RUNNING, "dispatched", owner=owner, attempts=n.attempts + 1)

    def complete(self, node_id: str, output: dict[str, Any], cost: dict[str, Any]) -> Node:
        n = self.set_status(node_id, NodeStatus.DONE, "verified output", output=output, cost=cost, error=None)
        self.refresh_ready()
        return n

    def fail(self, node_id: str, error: str, output: Optional[dict[str, Any]] = None, cost: Optional[dict[str, Any]] = None) -> Node:
        fields: dict[str, Any] = {"error": error[:2000]}
        if output is not None:
            fields["output"] = output
        if cost is not None:
            fields["cost"] = cost
        n = self.set_status(node_id, NodeStatus.FAILED, "failed", **fields)
        self.refresh_ready()
        return n

    def retry(self, node_id: str) -> Node:
        return self.set_status(node_id, NodeStatus.READY, "retry")

    def recover_running(self) -> list[str]:
        """After a crash: nodes left RUNNING go back to READY (their side effects are idempotent)."""
        out = []
        for n in self.nodes(NodeStatus.RUNNING):
            self.set_status(n.id, NodeStatus.READY, "recovered after restart")
            out.append(n.id)
        return out

    # -- rollback and replan ------------------------------------------------
    def mark_stale(self, node_id: str, reason: str = "upstream invalidated") -> list[str]:
        """The node and everything that depends on it become stale; unaffected branches are untouched (§11.6)."""
        affected = [node_id] + self.descendants(node_id)
        with self.db._lock:
            for nid in affected:
                n = self.node(nid)
                if n.status != NodeStatus.STALE:
                    self.set_status(nid, NodeStatus.STALE, reason)
        return affected

    def splice(self, subplan: Plan, replacing: Iterable[str], generation: int) -> list[str]:
        """Replace stale nodes with a revised sub-graph. History of replaced nodes is kept (status skipped)."""
        replacing = list(replacing)
        with self.db._lock:
            c = self.db.conn()
            old_dependents: dict[str, list[str]] = {r: self.dependents(r) for r in replacing}
            for r in replacing:
                n = self.node(r)
                if n.status in (NodeStatus.STALE, NodeStatus.FAILED, NodeStatus.BLOCKED, NodeStatus.PENDING, NodeStatus.READY):
                    self.set_status(r, NodeStatus.SKIPPED, f"replaced by plan generation {generation}")
                c.execute("DELETE FROM edges WHERE run_id=? AND (from_id=? )", (self.run_id, r))
            new_ids = []
            for s in subplan.steps:
                self._insert(s, generation)
                new_ids.append(s.id)
            for s in subplan.steps:
                for dep in s.depends_on:
                    c.execute("INSERT OR IGNORE INTO edges VALUES (?,?,?)", (self.run_id, dep, s.id))
            # Downstream nodes that depended on replaced nodes now depend on the new leaves.
            leaves = [s.id for s in subplan.steps if not any(s.id in t.depends_on for t in subplan.steps)]
            for r, deps in old_dependents.items():
                for d in deps:
                    if d in replacing or d in new_ids:
                        continue
                    for leaf in leaves:
                        c.execute("INSERT OR IGNORE INTO edges VALUES (?,?,?)", (self.run_id, leaf, d))
                    n = self.node(d)
                    if n.status in (NodeStatus.STALE, NodeStatus.BLOCKED):
                        self.set_status(d, NodeStatus.PENDING, "awaiting replanned upstream")
            self.refresh_ready()
        return new_ids

    def snapshot(self) -> dict[str, Any]:
        return {"nodes": [n.to_dict() for n in self.nodes()], "edges": [dict(r) for r in self.db.conn().execute("SELECT from_id, to_id FROM edges WHERE run_id=?", (self.run_id,)).fetchall()]}
