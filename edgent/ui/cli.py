"""The command line: init, chat, run, build, sessions, sources, outline, doctor, setup, mcp, serve, eval.

    edgent init [path]                 create a study workspace
    edgent chat                        a long-running conversation (REPL)
    edgent run "task"                  one turn, non-interactive, prints the answer
    edgent build "goal"                plan and build a book/course/assignment pack (resumable)
    edgent sources add <path|url>      ingest a source into greppable text with locators
    edgent doctor                      toolchains, provider, workspace
    edgent setup --host agy|claude     register Edgent as an MCP server + install skills for a host
    edgent mcp                         serve the tools over MCP (stdio)
    edgent serve                       the local web UI (reader + tutor + dashboards)
    edgent eval [dir]                  run the golden scenarios through the gate
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import webbrowser
from pathlib import Path
from typing import Any

from ..domain.errors import ConfigurationError
from ..domain.workspace import Workspace


def _app(args: argparse.Namespace, create: bool = False) -> Any:
    from ..application.app import App, AppConfig

    cfg = AppConfig(
        provider=getattr(args, "provider", None),
        model=getattr(args, "model", None),
        effort=getattr(args, "effort", None) or "medium",
        mode=getattr(args, "mode", None),
        verbose=bool(getattr(args, "verbose", False)),
    )
    return App.open(getattr(args, "workspace", None) or ".", cfg, create=create)


def _print_result(result: Any, as_json: bool) -> None:
    if as_json:
        print(json.dumps(result.to_dict() if hasattr(result, "to_dict") else result, indent=2, default=str))
        return
    print(result.text if hasattr(result, "text") else result)
    stop = getattr(result, "stop_reason", "done")
    if stop != "done":
        print(f"\n[stopped: {stop}]", file=sys.stderr)


def _console_ask_user(question: str) -> str:
    return input(f"\n[edgent asks] {question}\n> ")


def _console_ask_permission(spec: Any, args: dict[str, Any]) -> bool:
    shown = json.dumps(args, ensure_ascii=False)[:300]
    ans = input(f"\n[permission] allow {spec.name} ({spec.permission}) with {shown}? [y/N] ").strip().lower()
    return ans in ("y", "yes")


def _verbose_events(ev: dict[str, Any]) -> None:
    kind = ev.get("kind")
    if kind == "tool_call":
        print(f"  → {ev.get('name')} {json.dumps(ev.get('arguments', {}), ensure_ascii=False)[:160]}", file=sys.stderr)
    elif kind == "model_call":
        u = ev.get("usage") or {}
        print(f"  · model ({u.get('prompt_tokens', 0)}+{u.get('completion_tokens', 0)} tok)", file=sys.stderr)


# --------------------------------------------------------------------------- commands
def cmd_init(args: argparse.Namespace) -> int:
    ws = Workspace.init(args.path, name=args.name, git=not args.no_git)
    print(f"workspace ready at {ws.root}\n  edit STUDY.md, add sources with `edgent sources add`, then `edgent chat` or `edgent build`")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    app = _app(args)
    kw: dict[str, Any] = {"on_event": _verbose_events if args.verbose else None}
    if args.interactive:
        kw["ask_user"] = _console_ask_user
        kw["ask_permission"] = _console_ask_permission
    session, result = app.chat(args.task, session_id=args.session, **kw)
    _print_result(result, args.json)
    print(f"session: {session.session_id}", file=sys.stderr)
    return 0 if result.stop_reason in ("done", "max_iterations") else 1


def cmd_chat(args: argparse.Namespace) -> int:
    app = _app(args)
    session = app.load_session(args.session) if args.session else app.new_session("chat", title=args.title)
    loop = app.loop(session, ask_user=_console_ask_user, ask_permission=_console_ask_permission, on_event=_verbose_events if args.verbose else None)
    print(f"edgent chat · workspace {app.workspace.root.name} · session {session.session_id} · provider {app.provider.name}\n/help for commands, /exit to leave", file=sys.stderr)
    while True:
        try:
            text = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text in ("/exit", "/quit"):
            break
        if text == "/help":
            print("/exit  /todo  /stance  /compact  /sessions  /status")
            continue
        if text == "/todo":
            print(json.dumps(session.todo, indent=2))
            continue
        if text == "/stance":
            print("\n".join(session.stance) or "(empty)")
            continue
        if text == "/compact":
            rep = loop.compactor.maybe_compact(session, app.provider.caps.context_tokens, force=True)
            print(rep)
            continue
        if text == "/sessions":
            for s in app.sessions().list():
                print(f"{s['session_id']}  {s['mode']:9s} {s.get('title', '')}")
            continue
        if text == "/status":
            print(json.dumps(app.outline().summary(), indent=2))
            continue
        result = loop.run_turn(text, extra_context={"learner_summary": app.learner_summary()})
        print(f"\nedgent> {result.text}")
        if result.stop_reason != "done":
            print(f"[stopped: {result.stop_reason}]", file=sys.stderr)
    print(f"session saved: {session.session_id}", file=sys.stderr)
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    app = _app(args)
    on_event = _verbose_events if args.verbose else None
    outline = app.outline()
    if args.goal and not outline.units:
        print("planning the outline…", file=sys.stderr)
        app.plan_build(args.goal, on_event=on_event)
        outline = app.outline()
        print(outline.render_markdown(), file=sys.stderr)
        if args.plan_only:
            return 0
    elif not outline.units:
        print("no outline yet; give a goal: edgent build \"a 100-page book on …\"", file=sys.stderr)
        return 2
    job = app.build_job(on_event=on_event, max_parallel=args.parallel)
    report = job.run(max_units=args.max_units)
    print(json.dumps(report.to_dict() if hasattr(report, "to_dict") else report, indent=2, default=str))
    print(job.walkthrough() if hasattr(job, "walkthrough") else "")
    return 0


def cmd_sessions(args: argparse.Namespace) -> int:
    app = _app(args)
    rows = app.sessions().list(mode=args.mode)
    if args.json:
        print(json.dumps(rows, indent=2, default=str))
    else:
        for s in rows:
            print(f"{s['session_id']}  {s['mode']:9s} {s.get('created_at', '')[:19]}  {s.get('messages', 0):4d} msgs  {s.get('title', '')}")
    return 0


def cmd_trace(args: argparse.Namespace) -> int:
    from ..observability import TraceReader

    app = _app(args)
    reader = TraceReader(str(app.workspace.path("traces") / f"{args.session_id}.jsonl"))
    if args.summary:
        print(json.dumps(reader.summary(), indent=2))
        return 0
    for ev in reader:
        print(f"[{ev.seq:04d}] {ev.kind:12s} {ev.node_id or '':14s} {json.dumps(ev.data, default=str)[:200]}")
    return 0


def cmd_sources(args: argparse.Namespace) -> int:
    from ..adapters import ingest

    app = _app(args)
    if args.sources_cmd == "add":
        for item in args.items:
            if item.startswith(("http://", "https://")):
                rec = ingest.ingest_url(item, app.workspace, title=args.title) if "title" in ingest.ingest_url.__code__.co_varnames else ingest.ingest_url(item, app.workspace)
            else:
                rec = ingest.ingest_path(item, app.workspace, app.toolchain, title=args.title)
            d = rec.to_dict() if hasattr(rec, "to_dict") else rec
            print(f"added {d['id']} ({d['kind']}, coverage {d['coverage']['status']}) -> {d['text_path']}")
            for lim in d.get("limitations", []):
                print(f"  note: {lim}")
        return 0
    if args.sources_cmd == "list":
        for rec in ingest.list_sources(app.workspace):
            d = rec.to_dict() if hasattr(rec, "to_dict") else rec
            print(f"{d['id']:30s} {d['kind']:8s} coverage {d['coverage']['status']:8s} {d['title']}")
        return 0
    if args.sources_cmd == "cite":
        print(json.dumps(ingest.resolve_locator(app.workspace, args.locator), indent=2, default=str))
        return 0
    return 2


def cmd_outline(args: argparse.Namespace) -> int:
    app = _app(args)
    o = app.outline()
    print(json.dumps(o.to_dict(), indent=2) if args.json else o.render_markdown())
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    app = _app(args)
    print(json.dumps({"outline": app.outline().summary(), "unknowns_open": len(app.unknowns().open()), "progress_tail": app.workspace.progress_tail(10)}, indent=2))
    return 0


def cmd_walkthrough(args: argparse.Namespace) -> int:
    app = _app(args)
    print(app.unknowns().render_markdown())
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    try:
        app = _app(args)
    except FileNotFoundError:
        from ..adapters.toolchain import Toolchain

        report: dict[str, Any] = Toolchain.detect().report()
        report["workspace"] = None
    else:
        report = app.doctor()
        if args.host:
            from ..application import hosts

            report["host"] = hosts.doctor(args.host, app.workspace)
    if args.json:
        print(json.dumps(report, indent=2, default=str))
        return 0
    for item in report.get("items", []):
        mark = "ok     " if item["available"] else "missing"
        print(f"{mark} {item['name']:16s} {item.get('path') or ''}{('  ' + item['hint']) if item.get('hint') else ''}")
    prov = report.get("provider") or {}
    print(f"\nprovider: {prov.get('name')} {prov.get('model', '')}  health: {json.dumps(prov.get('health'), default=str)[:200]}")
    if report.get("workspace") is None:
        print("workspace: none here (run `edgent init`)")
    else:
        print(f"workspace: {report['workspace']['root']}  git: {report['workspace']['git']}")
    if "host" in report:
        print(f"host {args.host}: {json.dumps(report['host'], indent=2, default=str)}")
    return 0


def cmd_setup(args: argparse.Namespace) -> int:
    from ..application import hosts

    app = _app(args)
    changes = hosts.plan(args.host, app.workspace)
    for c in changes:
        print(f"{c.action:6s} {c.path}  {c.note}")
    if args.dry_run:
        return 0
    if not args.yes:
        ans = input("apply these changes (backups are kept)? [y/N] ").strip().lower()
        if ans not in ("y", "yes"):
            return 1
    written = hosts.apply(changes)
    hosts.install_skills(args.host, app.workspace)
    for w in written:
        print(f"wrote {w}")
    print(f"done. Restart {args.host} and run `edgent doctor --host {args.host}` to confirm.")
    return 0


def cmd_mcp(args: argparse.Namespace) -> int:
    from ..tools import mcp_prompts
    from ..tools.mcp_server import MCPServer

    app = _app(args)
    server = MCPServer(app.registry, lambda: app.tool_context(), prompts=getattr(mcp_prompts, "PROMPTS", None))
    server.serve(sys.stdin, sys.stdout)
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from .web.app_service import AppService
    from .web.server import LocalServer

    app = _app(args, create=False)
    service = AppService(app)
    token = None if args.no_token else LocalServer.new_token()
    server = LocalServer(service, port=args.port, token=token)
    print(f"Edgent UI at {server.url}", file=sys.stderr)
    if args.heartbeat:
        service.start_heartbeat()
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(server.url)).start()
    server.serve_forever()
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from ..evals.gate import load_scenarios, record_results
    from ..evals.runner import run_scenarios

    scenarios = load_scenarios(args.dir)
    if args.only:
        scenarios = [s for s in scenarios if s.id in args.only or any(t in args.only for t in s.tags)]
    results = run_scenarios(scenarios)
    for r in results:
        mark = "PASS" if r.passed else "FAIL"
        print(f"{mark} {r.scenario_id:36s} calls={r.tool_calls} tokens={r.prompt_tokens + r.completion_tokens} {'' if r.passed else ', '.join(r.failed_checks())}")
    row = record_results(results, Path(args.dir).parent / "results.md", args.label)
    print(row.strip())
    return 0 if all(r.passed for r in results) else 1


def cmd_heartbeat(args: argparse.Namespace) -> int:
    from ..application.heartbeat import Heartbeat

    app = _app(args)
    hb_settings = app.settings.get("heartbeat") or {}

    def run(prompt: str) -> Any:
        _, result = app.chat(prompt, on_event=_verbose_events if args.verbose else None)
        return result.text

    hb = Heartbeat(app.workspace, run, times=hb_settings.get("times"), interval_minutes=hb_settings.get("interval_minutes"), prompt=hb_settings.get("prompt"), tz=hb_settings.get("timezone", "UTC"))
    out = hb.tick() if not args.force else {"ran": True, "result": run(hb_settings.get("prompt") or hb.prompt)}
    print(json.dumps(out, indent=2, default=str))
    return 0


# --------------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="edgent", description="Edgent: a general-purpose agent harness with a deep education layer")
    ap.add_argument("--workspace", "-w", help="workspace directory (default: found from the current directory)")
    ap.add_argument("--provider", choices=["gemini", "openai", "agy", "claude", "mock"], help="override the configured provider")
    ap.add_argument("--model", help="override the model id")
    ap.add_argument("--effort", choices=["low", "medium", "high"], help="effort level")
    ap.add_argument("--mode", choices=["ask", "accept-edits", "auto", "plan", "bypass"], help="permission mode")
    ap.add_argument("-v", "--verbose", action="store_true", help="show tool calls and model calls as they happen")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("init", help="create a study workspace")
    p.add_argument("path", nargs="?", default=".")
    p.add_argument("--name")
    p.add_argument("--no-git", action="store_true")
    p.set_defaults(fn=cmd_init)

    p = sub.add_parser("run", help="one task, non-interactive; prints the answer")
    p.add_argument("task")
    p.add_argument("--session", help="continue an existing session")
    p.add_argument("--interactive", action="store_true", help="let the agent ask questions and request permissions on the console")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("chat", help="a long-running conversation")
    p.add_argument("--session")
    p.add_argument("--title")
    p.set_defaults(fn=cmd_chat)

    p = sub.add_parser("build", help="plan and build units (book, course, assignments); resumable")
    p.add_argument("goal", nargs="?")
    p.add_argument("--max-units", type=int)
    p.add_argument("--parallel", type=int, default=1)
    p.add_argument("--plan-only", action="store_true")
    p.set_defaults(fn=cmd_build)

    p = sub.add_parser("sessions", help="list sessions")
    p.add_argument("--mode")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_sessions)

    p = sub.add_parser("trace", help="print a session's trace")
    p.add_argument("session_id")
    p.add_argument("--summary", action="store_true")
    p.set_defaults(fn=cmd_trace)

    p = sub.add_parser("sources", help="manage sources")
    sp = p.add_subparsers(dest="sources_cmd", required=True)
    a = sp.add_parser("add")
    a.add_argument("items", nargs="+", help="files or URLs")
    a.add_argument("--title")
    sp.add_parser("list")
    c = sp.add_parser("cite")
    c.add_argument("locator")
    p.set_defaults(fn=cmd_sources)

    p = sub.add_parser("outline", help="show the outline")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_outline)
    sub.add_parser("status", help="build status").set_defaults(fn=cmd_status)
    sub.add_parser("walkthrough", help="the unknowns ledger as Markdown").set_defaults(fn=cmd_walkthrough)

    p = sub.add_parser("doctor", help="toolchains, provider, workspace, host registration")
    p.add_argument("--host", choices=["agy", "claude", "codex", "cursor", "generic"])
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_doctor)

    p = sub.add_parser("setup", help="register Edgent with a host (MCP server + skills)")
    p.add_argument("--host", required=True, choices=["agy", "claude", "codex", "cursor", "generic"])
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--yes", action="store_true")
    p.set_defaults(fn=cmd_setup)

    sub.add_parser("mcp", help="serve the tools over MCP on stdio").set_defaults(fn=cmd_mcp)

    p = sub.add_parser("serve", help="the local web UI")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-browser", action="store_true")
    p.add_argument("--no-token", action="store_true")
    p.add_argument("--heartbeat", action="store_true", help="run the heartbeat scheduler alongside the UI")
    p.set_defaults(fn=cmd_serve)

    p = sub.add_parser("eval", help="run golden scenarios through the gate")
    p.add_argument("dir", nargs="?", default="evals/golden")
    p.add_argument("--label", default="")
    p.add_argument("--only", nargs="*", help="scenario ids or tags")
    p.set_defaults(fn=cmd_eval)

    p = sub.add_parser("heartbeat", help="run one heartbeat tick")
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_heartbeat)
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    try:
        return int(args.fn(args) or 0)
    except ConfigurationError as e:
        print(f"configuration error: {e}", file=sys.stderr)
        return 2
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
