"""Command-line interface.

    python -m harness repl --workspace .
    python -m harness serve --workspace . --port 8765
    python -m harness chat "Inspect the repo and run tests" --workspace .
    python -m harness mcp --workspace .
    python -m harness doctor
    python -m harness run "What is the total on this invoice?" --file invoice.png
    python -m harness resume run_xxx
    python -m harness health | tools | runs | trace RUN_ID | eval golden.json | ablate golden.json
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
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
    if getattr(args, "allow_network", False):
        cfg.sandbox.allow_network = True
    if getattr(args, "python_bin", None):
        cfg.sandbox.python_executable = args.python_bin
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


def _add_session_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--workspace", default=".", help="persistent workspace directory")
    p.add_argument("--mode", choices=["ask", "accept_edits", "auto", "plan"], default="accept_edits")
    p.add_argument("--skills-dir", action="append", default=[], help="extra SKILL.md directory (repeatable)")
    p.add_argument("--agents-dir", action="append", default=[], help="extra subagent .md directory (repeatable)")
    p.add_argument("--context-file", action="append", default=[], help="extra workspace context file to inject (repeatable)")
    p.add_argument("--allow-network", action="store_true", help="allow outbound network access in sandboxed bash/python")
    p.add_argument("--python-bin", help="Python interpreter for sandboxed code execution")
    p.add_argument("--text-tools", action="store_true", help="use sentinel-safe text tool protocol (for non-native-tool bridges)")
    p.add_argument("--single-turn-bridge", action="store_true", help="flatten multi-turn history for single-turn CLI bridges")
    p.add_argument("--max-prompt-bytes", type=int, default=110_000, help="pre-flight prompt ceiling in bytes before compaction")
    p.add_argument("--mcp-server", action="append", default=[], help="attach stdio MCP server: 'name=cmd args...' (repeatable)")


def _build_client(args: argparse.Namespace, cfg: HarnessConfig):
    from .core.client import MockModelClient, OpenAICompatibleClient
    from .core.providers import AnthropicMessagesClient

    backend = getattr(args, "backend", "mock")
    if backend == "mock":
        return MockModelClient()
    if backend == "anthropic":
        return AnthropicMessagesClient(
            base_url=cfg.serving.base_url,
            model=cfg.serving.model,
            api_key=cfg.serving.api_key,
            effort=getattr(args, "global_effort", None),
            timeout_s=cfg.serving.timeout_s,
            max_retries=cfg.serving.max_retries,
            max_prompt_bytes=getattr(args, "max_prompt_bytes", None),
            native_tool_calls=not getattr(args, "text_tools", False),
            single_turn_flatten=getattr(args, "single_turn_bridge", False),
        )
    s, c = cfg.serving, cfg.capabilities
    return OpenAICompatibleClient(
        s.base_url, s.model, s.api_key, s.timeout_s, s.max_retries,
        s.price_per_million_prompt, s.price_per_million_completion,
        c.native_image, c.native_audio, c.native_video, c.thinking_switch,
    )


def _build_loop(args: argparse.Namespace):
    from .sandbox.workspace import Workspace
    from .session import AgentLoop, PermissionMode, PermissionPolicy, SkillRegistry, SubagentRegistry
    from .tools.mcp_client import MCPClient

    cfg = _config(args)
    ws = Workspace.open(args.workspace, create_subdirs=False)
    if not getattr(args, "state_dir", None):
        cfg.state_dir = os.path.join(ws.path, ".edgent")
    skills = SkillRegistry()
    for sd in getattr(args, "skills_dir", []):
        skills.load_dir(sd)
    subagents = SubagentRegistry()
    for ad in getattr(args, "agents_dir", []):
        subagents.load_dir(ad)

    loop = AgentLoop(
        workspace=ws,
        client=_build_client(args, cfg),
        config=cfg,
        skills=skills,
        subagents=subagents,
        policy=PermissionPolicy(mode=PermissionMode(args.mode)),
        native_tool_calls=not getattr(args, "text_tools", False),
        max_prompt_bytes=getattr(args, "max_prompt_bytes", 110_000),
        context_files=list(getattr(args, "context_file", [])),
    )
    for spec in getattr(args, "mcp_server", []):
        if "=" in spec:
            name, cmd_str = spec.split("=", 1)
        else:
            name, cmd_str = "ext", spec
        argv = shlex.split(cmd_str)
        if argv:
            loop.attach_mcp_client(MCPClient(argv, server_name=name.strip() or "ext", cwd=ws.path))
    return loop


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
    ap = argparse.ArgumentParser(prog="harness", description="Edgent general-purpose multimodal agent harness")
    ap.add_argument("--config", help="JSON config file")
    ap.add_argument("--state-dir", help="where state, artifacts, traces and memory live")
    ap.add_argument("--backend", choices=["mock", "openai", "anthropic"], default=os.environ.get("HARNESS_BACKEND", "mock"))
    ap.add_argument("--base-url")
    ap.add_argument("--model")
    ap.add_argument("--api-key")
    ap.add_argument("--effort", dest="global_effort")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("-v", "--verbose", action="store_true", help="stream trace events to stderr")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="run a structured DAG task")
    p_run.add_argument("task")
    p_run.add_argument("--file", "-f", action="append", default=[], help="input file (text, image, video, audio); repeatable")
    p_run.add_argument("--text", action="append", default=[], help="inline text input; repeatable")
    p_run.add_argument("--effort", choices=[e.value for e in EffortLevel])
    p_run.add_argument("--user", default="default")
    p_run.add_argument("--interactive", action="store_true", help="let the agent ask you questions")

    p_chat = sub.add_parser("chat", help="send one turn to a persistent general-purpose workspace session")
    p_chat.add_argument("message", nargs="?", default="", help="user message (optional with --approve/--reject/--undo)")
    p_chat.add_argument("--session", default="default", help="session id")
    p_chat.add_argument("--approve", action="store_true", help="approve a pending review and resume")
    p_chat.add_argument("--reject", action="store_true", help="reject a pending review with message as feedback")
    p_chat.add_argument("--undo", type=int, default=0, help="rewind workspace edits from the last N turns")
    p_chat.add_argument("--stream", action="store_true", help="stream model text deltas live to stdout")
    _add_session_flags(p_chat)

    p_repl = sub.add_parser("repl", help="launch the interactive streaming terminal REPL")
    p_repl.add_argument("--session", default="default", help="initial session id")
    p_repl.add_argument("--show-thinking", action="store_true", help="print model reasoning deltas to stderr")
    _add_session_flags(p_repl)

    p_serve = sub.add_parser("serve", help="run the local HTTP + SSE streaming agent daemon")
    p_serve.add_argument("--host", default="127.0.0.1", help="bind host (default 127.0.0.1)")
    p_serve.add_argument("--port", type=int, default=8765, help="bind port (default 8765)")
    p_serve.add_argument("--auth-token", default="", help="optional Bearer token required on all requests")
    p_serve.add_argument("--heartbeat", help="optional JSON file of periodic HeartbeatJob definitions")
    _add_session_flags(p_serve)

    p_mcp = sub.add_parser("mcp", help="run a stdio MCP server exposing workspace tools and prompts")
    p_mcp.add_argument("--workspace", default=".", help="persistent workspace directory")
    p_mcp.add_argument("--mode", choices=["ask", "accept_edits", "auto", "plan"], default="accept_edits")
    p_mcp.add_argument("--compact", action="store_true", help="serve compact tool schemas")

    sub.add_parser("doctor", help="read-only check of binaries, sandbox isolation, and environment")

    p_res = sub.add_parser("resume", help="resume an interrupted DAG run")
    p_res.add_argument("run_id", nargs="?", help="omit to resume every in-flight run")

    sub.add_parser("health", help="model server, sidecars and state health")
    sub.add_parser("tools", help="MCP-shaped tool manifest")
    sub.add_parser("runs", help="list DAG runs")
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
    if args.cmd == "doctor":
        from .session.doctor import run_doctor

        print(json.dumps(run_doctor(), indent=2))
        return 0

    if args.cmd == "chat":
        loop = _build_loop(args)
        try:
            if args.undo > 0:
                out = loop.undo(args.session, steps=args.undo)
                print(json.dumps(out, indent=2))
                return 0 if out.get("ok") else 1
            streamed = False

            def _on_ev(ev) -> None:
                nonlocal streamed
                if ev.kind == "text_delta" and ev.text:
                    streamed = True
                    sys.stdout.write(ev.text)
                    sys.stdout.flush()

            cb = _on_ev if (args.stream and not args.json) else None
            if args.approve or args.reject:
                res = loop.resume_review(args.session, approved=bool(args.approve), feedback=args.message, on_event=cb)
            else:
                res = loop.send(args.session, args.message, on_event=cb)
        finally:
            loop.close()

        if args.json:
            print(json.dumps(res.to_dict(), indent=2, default=str))
        else:
            if streamed:
                print()
            else:
                print(f"[{res.status}] session={res.session_id} turns={res.turns_used} tools={res.tool_calls_used}\n\n{res.reply}")
            if res.review_request:
                print("\nPENDING REVIEW:", json.dumps(res.review_request, indent=2))
        return 0 if res.status in ("completed", "awaiting_review") else 1

    if args.cmd == "repl":
        from .ui.repl import run_repl

        loop = _build_loop(args)
        try:
            return run_repl(loop, session_id=args.session, show_thinking=args.show_thinking)
        finally:
            loop.close()

    if args.cmd == "serve":
        from .session.heartbeat import HeartbeatScheduler
        from .ui.server import HarnessServer

        loop = _build_loop(args)
        hb = None
        if args.heartbeat:
            hb = HeartbeatScheduler(
                lambda sid, prompt, stg: loop.send(sid, prompt, stage=stg),
                log_path=os.path.join(loop.config.state_dir, "heartbeat.jsonl"),
            )
            hb.load_jobs_file(args.heartbeat)
        srv = HarnessServer(loop, host=args.host, port=args.port, auth_token=args.auth_token, heartbeat=hb)
        print(f"Edgent HTTP/SSE server listening on {srv.base_url} (workspace={loop.workspace.path})", file=sys.stderr)
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            srv.stop()
            loop.close()
        return 0

    if args.cmd == "mcp":
        from .modality.artifacts import ArtifactStore
        from .modality.sidecars import Sidecars
        from .sandbox.workspace import Workspace
        from .session.policy import PermissionMode, PermissionPolicy
        from .tools import IdempotencyStore, MCPServer, ToolContext, build_default_registry, register_general_tools

        cfg = _config(args)
        ws = Workspace.open(args.workspace, create_subdirs=False)
        if not getattr(args, "state_dir", None):
            cfg.state_dir = os.path.join(ws.path, ".edgent")
        state_dir = cfg.state_dir
        os.makedirs(state_dir, exist_ok=True)
        store = ArtifactStore(os.path.join(state_dir, "artifacts"))
        reg = register_general_tools(build_default_registry(IdempotencyStore(os.path.join(state_dir, "state.sqlite3"))))
        policy = PermissionPolicy(mode=PermissionMode(args.mode))
        srv = MCPServer(
            reg,
            lambda: ToolContext("mcp", "mcp", store, ws, Sidecars(), cfg, extra={"policy": policy}),
            compact_schemas=args.compact,
        )
        try:
            srv.serve_stdio()
        finally:
            store.close()
            reg._pool.shutdown(wait=False)
        return 0

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
