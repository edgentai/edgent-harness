"""Command-line interface.

    python -m harness run "What is the total on this invoice?" --file invoice.png
    python -m harness run "..." --backend openai --base-url http://127.0.0.1:8000/v1 --model Qwen3.8-27B
    python -m harness resume run_xxx
    python -m harness health | tools | runs | trace RUN_ID | eval golden.json | ablate golden.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Optional

from .core.config import EffortLevel, HarnessConfig
from .orchestration.master import Harness


def _config(args: argparse.Namespace) -> HarnessConfig:
    cfg = HarnessConfig.load(getattr(args, "config", None))
    if getattr(args, "state_dir", None):
        cfg.state_dir = args.state_dir
    if getattr(args, "base_url", None):
        cfg.serving.base_url = args.base_url
    if getattr(args, "model", None):
        cfg.serving.model = args.model
    if getattr(args, "api_key", None):
        cfg.serving.api_key = args.api_key
    return cfg


def _harness(args: argparse.Namespace) -> Harness:
    cfg = _config(args)
    echo = None
    if getattr(args, "verbose", False):
        def echo(ev: Any) -> None:  # noqa: E306
            print(f"[{ev.seq:04d}] {ev.kind:10s} {ev.node_id or '':12s} {json.dumps(ev.data, default=str)[:160]}", file=sys.stderr)
    ask = (lambda q: input(f"\n[harness asks] {q}\n> ")) if getattr(args, "interactive", False) else None
    h = Harness(cfg, backend=getattr(args, "backend", "mock"), ask_user=ask, echo=echo)
    _OPEN.append(h)
    return h


def _print_result(r: Any, as_json: bool) -> None:
    if as_json:
        print(json.dumps(r.to_dict(), indent=2, default=str))
        return
    print(f"run: {r.run_id}\nstatus: {r.status}   effort: {r.effort}   type: {r.problem_type}   replans: {r.replans}   escalations: {r.escalations}")
    print(f"\nANSWER:\n{r.answer if isinstance(r.answer, str) else json.dumps(r.answer, indent=2, default=str)}")
    if r.claims:
        print("\nCLAIMS:")
        for c in r.claims:
            cites = ", ".join(f"{e['artifact_id'][:12]}:{e.get('kind')}:{e.get('locator')}" for e in c.get("evidence", []))
            print(f"  - [{'✓' if c.get('verified') else '?'}] {c['text']}  <{cites}>")
    if r.verdict:
        print(f"\nVERDICT: {r.verdict['status']} — {r.verdict.get('summary', '')}")
        for ch in r.verdict.get("checks", []):
            print(f"  {'PASS' if ch['passed'] else 'FAIL'} {ch['name']}: {ch.get('detail', '')[:120]}")
    if r.gaps:
        print("\nUNVERIFIED / GAPS:")
        for g in r.gaps:
            print(f"  - {g}")
    tot = (r.cost or {}).get("total", {})
    print(f"\ncost: {tot.get('prompt_tokens', 0)} prompt + {tot.get('completion_tokens', 0)} completion tokens, ${tot.get('cost_usd', 0):.4f}; wall {r.wall_clock_s:.1f}s")
    print(f"trace: {r.trace_path}")
    if r.message:
        print(f"note: {r.message}")


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="harness", description="Edgent general-purpose multimodal harness")
    ap.add_argument("--config", help="JSON config file")
    ap.add_argument("--state-dir", help="where state, artifacts, traces and memory live")
    ap.add_argument("--backend", choices=["mock", "openai"], default=os.environ.get("HARNESS_BACKEND", "mock"))
    ap.add_argument("--base-url")
    ap.add_argument("--model")
    ap.add_argument("--api-key")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("-v", "--verbose", action="store_true", help="stream trace events to stderr")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="run a task")
    p_run.add_argument("task")
    p_run.add_argument("--file", "-f", action="append", default=[], help="input file (text, image, video, audio); repeatable")
    p_run.add_argument("--text", action="append", default=[], help="inline text input; repeatable")
    p_run.add_argument("--effort", choices=[e.value for e in EffortLevel])
    p_run.add_argument("--user", default="default")
    p_run.add_argument("--interactive", action="store_true", help="let the agent ask you questions")

    p_res = sub.add_parser("resume", help="resume an interrupted run")
    p_res.add_argument("run_id", nargs="?", help="omit to resume every in-flight run")

    sub.add_parser("health", help="model server, sidecars and state health")
    sub.add_parser("tools", help="MCP-shaped tool manifest")
    sub.add_parser("runs", help="list runs")
    p_tr = sub.add_parser("trace", help="print a run's trace")
    p_tr.add_argument("run_id")
    p_tr.add_argument("--kind", help="filter by event kind")
    p_ev = sub.add_parser("eval", help="run a golden set")
    p_ev.add_argument("golden")
    p_ev.add_argument("--results", default="evals/results.md")
    p_ab = sub.add_parser("ablate", help="run ablations over a golden set")
    p_ab.add_argument("golden")
    p_ab.add_argument("--only", action="append")
    p_ab.add_argument("--results", default="evals/results.md")
    p_mem = sub.add_parser("memory", help="inspect or delete a user's memories")
    p_mem.add_argument("action", choices=["list", "search", "forget"])
    p_mem.add_argument("arg", nargs="?")
    p_mem.add_argument("--user", default="default")

    args = ap.parse_args(argv)
    try:
        return _dispatch(args)
    finally:
        for h in _OPEN:
            h.close()
        _OPEN.clear()


_OPEN: list[Harness] = []


def _dispatch(args: argparse.Namespace) -> int:
    if args.cmd == "run":
        h = _harness(args)
        r = h.run(args.task, files=args.file, effort=EffortLevel(args.effort) if args.effort else None, user_id=args.user, inline_texts=args.text)
        _print_result(r, args.json)
        return 0 if r.status in ("verified", "partially_verified") else 1
    if args.cmd == "resume":
        h = _harness(args)
        results = [h.resume(args.run_id)] if args.run_id else h.resume_all()
        for r in results:
            _print_result(r, args.json)
        return 0
    if args.cmd == "health":
        h = _harness(args)
        print(json.dumps(h.health(), indent=2, default=str))
        return 0
    if args.cmd == "tools":
        h = _harness(args)
        print(json.dumps(h.registry.mcp_manifest(), indent=2))
        return 0
    if args.cmd == "runs":
        h = _harness(args)
        for r in h.db.runs():
            print(f"{r['run_id']}  {r['status']:20s} {r.get('effort') or '':8s} {r['created_at']}  {r['task'][:60]}")
        return 0
    if args.cmd == "trace":
        from .observability.tracing import TraceReader

        cfg = _config(args)
        path = os.path.join(cfg.traces_dir, f"{args.run_id}.jsonl")
        reader = TraceReader(path)
        if args.json:
            print(json.dumps([e.to_dict() for e in reader if not args.kind or e.kind == args.kind], indent=1, default=str))
        else:
            for e in reader:
                if args.kind and e.kind != args.kind:
                    continue
                print(f"[{e.seq:04d}] {e.kind:10s} {e.node_id or '':14s} {json.dumps(e.data, default=str)[:200]}")
            print(json.dumps(reader.summary()))
        return 0
    if args.cmd == "eval":
        from .evals.golden import EvalRunner, GoldenSet

        h = _harness(args)
        summary = EvalRunner(h.run, args.results).run(GoldenSet.load(args.golden))
        print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2))
        for r in summary["results"]:
            print(f"  {'PASS' if r['passed'] else 'FAIL'} {r['case_id']:30s} {r['status']:20s} {r['detail']}")
        return 0 if summary["accuracy"] == 1.0 else 1
    if args.cmd == "ablate":
        from .evals.ablation import run_ablations
        from .evals.golden import GoldenSet

        cfg = _config(args)
        def make(c: HarnessConfig) -> Harness:
            h = Harness(c, backend=args.backend)
            _OPEN.append(h)
            return h

        out = run_ablations(make, GoldenSet.load(args.golden), cfg, args.only, args.results)
        for name, s in out.items():
            print(f"{name:18s} accuracy={s['accuracy']:.2f} verified={s['verified_rate']:.2f} tokens={s['mean_tokens']:.0f}")
        return 0
    if args.cmd == "memory":
        h = _harness(args)
        if args.action == "list":
            for f in h.memory.semantic.facts(user_id=args.user):
                print(json.dumps(f.to_dict(), default=str))
        elif args.action == "search":
            print(json.dumps(h.memory.search(args.arg or "", k=10), indent=1, default=str))
        elif args.action == "forget":
            print(f"deleted {h.memory.forget_user(args.user)} memory records for user {args.user}")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
