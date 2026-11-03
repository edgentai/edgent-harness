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

Global options (--workspace, --provider, --model, --effort, --mode, -v) are accepted before or after the command.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import webbrowser
from pathlib import Path
from typing import Any

from ..domain.errors import ConfigurationError
from ..domain.outline import OutlineError
from ..domain.workspace import Workspace

PROVIDERS = ("gemini", "openai", "agy", "claude", "mock")
EFFORTS = ("low", "medium", "high")
MODES = ("ask", "accept-edits", "auto", "plan", "bypass")
HOSTS = ("agy", "claude", "codex", "cursor", "generic")
PREVIEW_CHARS = 160
RUN_DEFAULT_MODE = "accept-edits"  # `edgent run` without --interactive has no one to ask; writes inside the workspace are allowed


def _app(args: argparse.Namespace, create: bool = False) -> Any:
    from ..application.app import App, AppConfig

    cfg = AppConfig(
        provider=getattr(args, "provider", None),
        model=getattr(args, "model", None),
        effort=getattr(args, "effort", None) or "medium",
        mode=getattr(args, "mode", None),
        verbose=bool(getattr(args, "verbose", False)),
        critic=not getattr(args, "no_critic", False),
    )
    app = App.open(getattr(args, "workspace", None) or ".", cfg, create=create)
    meta = app.provider_meta
    if meta.get("warning"):
        print(f"warning: {meta['warning']}; answers come from the offline mock provider. Run `edgent doctor` for the provider options.", file=sys.stderr)
    for warning in meta.get("warnings", []):
        print(f"warning: {warning}", file=sys.stderr)
    return app


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
    kind = ev.get("type") or ev.get("event") or ev.get("kind")
    if kind == "tool_call":
        state = "ok" if ev.get("ok") else "failed"
        print(f"  -> {ev.get('name')} {state} ({ev.get('duration_s', 0)} s) {str(ev.get('preview', ''))[:PREVIEW_CHARS]}", file=sys.stderr)
    elif kind == "model_call":
        u = ev.get("usage") or {}
        tools = ", ".join(ev.get("tool_calls") or []) or "none"
        print(f"  .. model call {ev.get('iteration', '?')}: {u.get('prompt_tokens', 0)}+{u.get('completion_tokens', 0)} tokens; tools: {tools}", file=sys.stderr)
    elif kind == "compaction":
        print(f"  .. compaction: {json.dumps({k: v for k, v in ev.items() if k not in ('type', 'session_id')}, default=str)[:PREVIEW_CHARS]}", file=sys.stderr)
    elif kind in ("plan", "unit_start", "unit_finished", "unit_blocked", "critic", "critic_error", "critic_exhausted", "fix_unit", "resumed", "run_finished"):
        detail = ev.get("status") or ev.get("reason") or ev.get("error") or ""
        print(f"  == {kind} {ev.get('unit_id', '')} {detail}".rstrip(), file=sys.stderr)


def _attachments(paths: list[str] | None) -> list[Any]:
    from ..domain.types import ContentPart

    parts = []
    for p in paths or []:
        path = Path(p)
        if not path.exists():
            raise FileNotFoundError(f"attachment not found: {path}")
        kind = "image" if path.suffix.lower() in (".png", ".jpg", ".jpeg", ".gif", ".webp") else "file"
        parts.append(ContentPart.of_media(kind, str(path.resolve())))
    return parts


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
    elif not getattr(args, "mode", None) and "mode" not in app.settings:
        kw["permission_mode"] = RUN_DEFAULT_MODE  # nobody can answer a prompt: workspace writes go ahead, commands and the network do not
    session, result = app.chat(args.task, session_id=args.session, parts=_attachments(args.attach), **kw)
    _print_result(result, args.json)
    print(f"session: {session.session_id}", file=sys.stderr)
    return 0 if result.stop_reason in ("done", "max_iterations") else 1


def cmd_chat(args: argparse.Namespace) -> int:
    app = _app(args)
    session = app.load_session(args.session) if args.session else app.new_session("chat", title=args.title)
    loop = app.loop(session, ask_user=_console_ask_user, ask_permission=_console_ask_permission, on_event=_verbose_events if args.verbose else None)
    print(f"edgent chat . workspace {app.workspace.root.name} . session {session.session_id} . provider {app.provider.name} {getattr(app.provider, 'model', '')}\n/help for commands, /exit to leave", file=sys.stderr)
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
            print("/exit  /todo  /stance  /compact  /sessions  /status  /recall <pattern>")
            continue
        if text == "/todo":
            print(json.dumps(session.todo, indent=2))
            continue
        if text == "/stance":
            print("\n".join(session.stance) or "(empty)")
            continue
        if text == "/compact":
            print(json.dumps(loop.compactor.maybe_compact(session, app.provider.caps.context_tokens, force=True).to_dict(), indent=2))
            continue
        if text == "/sessions":
            for s in app.sessions().list():
                print(f"{s['session_id']}  {s['mode']:9s} {s.get('title', '')}")
            continue
        if text == "/status":
            print(json.dumps(app.outline().summary(), indent=2))
            continue
        if text.startswith("/recall "):
            try:
                hits = session.recall(text[len("/recall "):].strip())
            except (ValueError, re.error) as e:
                print(f"recall failed: {e}", file=sys.stderr)
                continue
            for hit in hits:
                print(json.dumps(hit, default=str)[:300])
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
    if args.unblock:
        job = app.build_job(on_event=on_event)
        for uid in args.unblock:
            try:
                unit = job.unblock(uid)
            except OutlineError as e:
                print(f"cannot unblock {uid}: {e}", file=sys.stderr)
                return 2
            print(f"unit {unit.id}: {unit.status}", file=sys.stderr)
    if args.goal and not outline.units:
        print("planning the outline...", file=sys.stderr)
        try:
            app.plan_build(args.goal, on_event=on_event)
        except ValueError as e:
            print(f"planning failed: {e}\nThe initializer session is saved (edgent sessions); refine the goal or add sources and try again.", file=sys.stderr)
            return 2
        outline = app.outline()
        print(outline.render_markdown(), file=sys.stderr)
    elif not outline.units:
        print('no outline yet; give a goal: edgent build "a 100-page book on ..."', file=sys.stderr)
        return 2
    elif args.goal:
        print(f"an outline with {len(outline.units)} units exists; resuming it (the goal is kept for the record)", file=sys.stderr)
    if args.plan_only:
        if not args.goal:
            print(outline.render_markdown(), file=sys.stderr)
        return 0
    if not args.yes and sys.stdin.isatty():  # a person at the keyboard approves the plan before the workers start
        ans = input(f"run the build over {len(outline.units)} units now? [Y/n] ").strip().lower()
        if ans not in ("", "y", "yes"):
            print("not started; the outline is saved. Edit it with `edgent outline` in mind and run `edgent build` again.", file=sys.stderr)
            return 1
    job = app.build_job(on_event=on_event, max_parallel=args.parallel, max_rounds=args.rounds)
    try:
        report = job.run(max_units=args.max_units, stop_on_error=args.stop_on_error)
    except RuntimeError as e:  # another build holds the lock
        print(f"build not started: {e}", file=sys.stderr)
        return 2
    print(json.dumps(report.to_dict(), indent=2, default=str))
    print(job.walkthrough(), file=sys.stderr)
    return 0 if report.ok else 1


def cmd_sessions(args: argparse.Namespace) -> int:
    app = _app(args)
    rows = app.sessions().list(mode=args.kind)
    if args.json:
        print(json.dumps(rows, indent=2, default=str))
    else:
        for s in rows:
            print(f"{s['session_id']}  {s['mode']:9s} {str(s.get('created_at', ''))[:19]}  {int(s.get('messages', 0)):4d} msgs  {s.get('title', '')}")
    return 0


def cmd_trace(args: argparse.Namespace) -> int:
    from ..observability import TraceReader

    app = _app(args)
    path = app.workspace.path("traces") / f"{args.session_id}.jsonl"
    if not path.exists():
        print(f"no trace for session {args.session_id!r} (looked at {path}); `edgent sessions` lists the ids", file=sys.stderr)
        return 2
    reader = TraceReader(str(path))
    if args.summary:
        print(json.dumps(reader.summary(), indent=2))
        return 0
    for ev in reader:
        print(f"[{ev.seq:04d}] {ev.kind:12s} {ev.node_id or '':14s} {json.dumps(ev.data, default=str)[:200]}")
    return 0


def cmd_sources(args: argparse.Namespace) -> int:
    from ..tools.registry import AllowAll

    app = _app(args)
    ctx = app.tool_context(role="cli", permissions=AllowAll())
    if args.sources_cmd == "add":
        rc = 0
        for item in args.items:
            is_url = item.startswith(("http://", "https://"))
            call_args: dict[str, Any] = {"url" if is_url else "path": item}
            if args.title:
                call_args["title"] = args.title
            if args.replace:
                call_args["replace"] = True
            res = app.registry.call("sources_add_url" if is_url else "sources_add", call_args, ctx)
            if not res.ok:
                err = res.error.to_dict() if res.error else {}
                print(f"failed {item}: {err.get('message', 'unknown error')}" + (f"\n  {err['remediation']}" if err.get("remediation") else ""), file=sys.stderr)
                rc = 1
                continue
            d = res.output if isinstance(res.output, dict) else {}
            cov = d.get("coverage") if isinstance(d.get("coverage"), dict) else {}
            print(f"added {d.get('id', '?')} ({d.get('kind', '?')}, coverage {cov.get('status', d.get('coverage', '?'))}) -> {d.get('text_path', '')}")
            for lim in d.get("limitations", []) or []:
                print(f"  note: {lim}")
        return rc
    if args.sources_cmd == "list":
        res = app.registry.call("sources_list", {}, ctx)
        rows = (res.output or {}).get("sources", []) if res.ok and isinstance(res.output, dict) else []
        if args.json:
            print(json.dumps(res.output if res.ok else res.error.to_dict(), indent=2, default=str))
            return 0 if res.ok else 1
        for d in rows:
            cov = d.get("coverage") if isinstance(d.get("coverage"), dict) else {"status": d.get("coverage")}
            print(f"{str(d.get('id', '')):30s} {str(d.get('kind', '')):8s} coverage {str(cov.get('status', '')):8s} {d.get('title', '')}")
        if not rows:
            print("no sources yet; add one with `edgent sources add <file or url>`", file=sys.stderr)
        return 0
    if args.sources_cmd == "cite":
        res = app.registry.call("sources_cite_check", {"locator": args.locator}, ctx)
        print(json.dumps(res.output if res.ok else res.error.to_dict(), indent=2, default=str))
        return 0 if res.ok else 1
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


def cmd_skills(args: argparse.Namespace) -> int:
    app = _app(args)
    index = app.agents if args.kind == "agents" else app.skills
    for name in index.names():
        item = index.get(name)
        print(f"{name:22s} [{getattr(item, 'source', '')}] {getattr(item, 'description', '')[:100]}")
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
    print(f"\nprovider: {prov.get('kind') or prov.get('name')} {prov.get('model', '')} (chosen by {prov.get('source', '?')})  health: {json.dumps(prov.get('health'), default=str)[:200]}")
    for w in [prov.get("warning"), *(prov.get("warnings") or [])]:
        if w:
            print(f"  warning: {w}")
    if report.get("workspace") is None:
        print("workspace: none here (run `edgent init`)")
    else:
        print(f"workspace: {report['workspace']['root']}  git: {report['workspace']['git']}  skills: {len(report.get('skills', []))}  agents: {len(report.get('agents', []))}  tools: {report.get('tools', 0)}")
    if "host" in report:
        print(f"host {args.host}: {json.dumps(report['host'], indent=2, default=str)}")
    return 0


def cmd_setup(args: argparse.Namespace) -> int:
    from ..application import hosts

    app = _app(args)
    changes = hosts.plan(args.host, app.workspace) + hosts.plan_skills(args.host, app.workspace)
    for c in changes:
        print(f"{c.action:6s} {c.path}  {c.note}")
    if args.dry_run:
        return 0
    if not args.yes:
        ans = input("apply these changes (backups are kept)? [y/N] ").strip().lower()
        if ans not in ("y", "yes"):
            return 1
    written = hosts.apply(changes)
    for w in written:
        print(f"wrote {w}")
    print(f"done. Restart {args.host} and run `edgent doctor --host {args.host}` to confirm.")
    return 0


def cmd_mcp(args: argparse.Namespace) -> int:
    from ..tools.mcp_server import MCPServer

    app = _app(args)
    server = MCPServer(app.registry, lambda: app.tool_context(role="host"))
    return int(server.serve() or 0)


def cmd_serve(args: argparse.Namespace) -> int:
    from .web.app_service import AppService
    from .web.server import LocalServer

    app = _app(args, create=False)
    service = AppService(app)
    token = None if args.no_token else LocalServer.new_token()
    server = LocalServer(service, port=args.port, token=token)
    print(f"Edgent UI at {server.url}", file=sys.stderr)
    if args.heartbeat and not service.start_heartbeat():
        print('no heartbeat configured: add {"heartbeat": {"times": ["07:30"], "prompt": "..."}} to .edgent/settings.json', file=sys.stderr)
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(server.url)).start()
    try:
        server.serve_forever()
    finally:
        service.stop_heartbeat()
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from ..evals.gate import load_scenarios, record_results
    from ..evals.runner import run_scenarios

    scenarios = load_scenarios(args.dir)
    if args.only:
        scenarios = [s for s in scenarios if s.id in args.only or any(t in args.only for t in s.tags)]
    if not scenarios:
        print(f"no scenarios found in {args.dir}", file=sys.stderr)
        return 2
    results = run_scenarios(scenarios)
    for r in results:
        mark = "PASS" if r.passed else "FAIL"
        print(f"{mark} {r.scenario_id:36s} calls={r.tool_calls} tokens={r.prompt_tokens + r.completion_tokens} {'' if r.passed else ', '.join(r.failed_checks())}")
    row = record_results(results, Path(args.dir).parent / "results.md", args.label)
    print(row.strip())
    return 0 if all(r.passed for r in results) else 1


def cmd_heartbeat(args: argparse.Namespace) -> int:
    app = _app(args)
    hb = app.heartbeat(on_event=_verbose_events if args.verbose else None)
    if not hb.configured() and not args.force:
        print("no heartbeat configured; add {\"heartbeat\": {\"times\": [\"07:30\"], \"prompt\": \"...\"}} to .edgent/settings.json", file=sys.stderr)
        return 2
    if args.force:
        prompt = hb.prompt if getattr(hb, "prompt", None) else "Run the configured heartbeat: review progress.md and the learner's due items, then write a short plan for today."
        out: dict[str, Any] = {"ran": True, "forced": True, "result": hb.run(prompt)}
    else:
        out = hb.tick()
    print(json.dumps(out, indent=2, default=str))
    return 0


# --------------------------------------------------------------------------- parser
def _global_options(parser: argparse.ArgumentParser, suppress: bool) -> None:
    """The same options on the top level and on every subcommand; SUPPRESS keeps a subcommand from overriding a top-level value."""
    d: Any = argparse.SUPPRESS if suppress else None
    parser.add_argument("--workspace", "-w", default=d, help="workspace directory (default: found from the current directory)")
    parser.add_argument("--provider", choices=PROVIDERS, default=d, help="override the configured provider")
    parser.add_argument("--model", default=d, help="override the model id")
    parser.add_argument("--effort", choices=EFFORTS, default=d, help="effort level")
    parser.add_argument("--mode", choices=MODES, default=d, help="permission mode")
    parser.add_argument("-v", "--verbose", action="store_true", default=argparse.SUPPRESS if suppress else False, help="show tool calls and model calls as they happen")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="edgent", description="Edgent: a general-purpose agent harness with a deep education layer")
    _global_options(ap, suppress=False)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name: str, help_text: str, fn: Any) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help_text)
        _global_options(p, suppress=True)
        p.set_defaults(fn=fn)
        return p

    p = add("init", "create a study workspace", cmd_init)
    p.add_argument("path", nargs="?", default=".")
    p.add_argument("--name")
    p.add_argument("--no-git", action="store_true")

    p = add("run", "one task, non-interactive; prints the answer", cmd_run)
    p.add_argument("task")
    p.add_argument("--session", help="continue an existing session")
    p.add_argument("--attach", action="append", help="attach an image or file to the message (repeatable)")
    p.add_argument("--interactive", action="store_true", help="let the agent ask questions and request permissions on the console")
    p.add_argument("--json", action="store_true")

    p = add("chat", "a long-running conversation", cmd_chat)
    p.add_argument("--session")
    p.add_argument("--title")

    p = add("build", "plan and build units (book, course, assignments); resumable", cmd_build)
    p.add_argument("goal", nargs="?")
    p.add_argument("--max-units", type=int)
    p.add_argument("--parallel", type=int, default=1)
    p.add_argument("--rounds", type=int, default=2, help="fix rounds per unit after the critic (max 5)")
    p.add_argument("--plan-only", action="store_true", help="plan (or show) the outline and stop")
    p.add_argument("--yes", "-y", action="store_true", help="start the workers without asking")
    p.add_argument("--no-critic", action="store_true", help="skip the separate-context critic after units become done")
    p.add_argument("--stop-on-error", action="store_true")
    p.add_argument("--unblock", action="append", metavar="UNIT", help="reset a blocked unit to pending before running (repeatable)")

    p = add("sessions", "list sessions", cmd_sessions)
    p.add_argument("--kind", choices=["chat", "build"], help="only sessions of this kind")
    p.add_argument("--json", action="store_true")

    p = add("trace", "print a session's trace", cmd_trace)
    p.add_argument("session_id")
    p.add_argument("--summary", action="store_true")

    p = add("sources", "manage sources", cmd_sources)
    sp = p.add_subparsers(dest="sources_cmd", required=True)
    a = sp.add_parser("add")
    a.add_argument("items", nargs="+", help="files or URLs")
    a.add_argument("--title")
    a.add_argument("--replace", action="store_true", help="re-ingest even if the content was added before")
    ls = sp.add_parser("list")
    ls.add_argument("--json", action="store_true")
    c = sp.add_parser("cite")
    c.add_argument("locator")

    p = add("outline", "show the outline", cmd_outline)
    p.add_argument("--json", action="store_true")
    add("status", "build status", cmd_status)
    add("walkthrough", "the unknowns ledger as Markdown", cmd_walkthrough)
    p = add("skills", "list the skills (or agents) visible to this workspace", cmd_skills)
    p.add_argument("kind", nargs="?", choices=["skills", "agents"], default="skills")

    p = add("doctor", "toolchains, provider, workspace, host registration", cmd_doctor)
    p.add_argument("--host", choices=HOSTS)
    p.add_argument("--json", action="store_true")

    p = add("setup", "register Edgent with a host (MCP server + skills)", cmd_setup)
    p.add_argument("--host", required=True, choices=HOSTS)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--yes", action="store_true")

    add("mcp", "serve the tools over MCP on stdio", cmd_mcp)

    p = add("serve", "the local web UI", cmd_serve)
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-browser", action="store_true")
    p.add_argument("--no-token", action="store_true")
    p.add_argument("--heartbeat", action="store_true", help="run the heartbeat scheduler alongside the UI")

    p = add("eval", "run golden scenarios through the gate", cmd_eval)
    p.add_argument("dir", nargs="?", default="evals/golden")
    p.add_argument("--label", default="")
    p.add_argument("--only", nargs="*", help="scenario ids or tags")

    p = add("heartbeat", "run one heartbeat tick", cmd_heartbeat)
    p.add_argument("--force", action="store_true", help="run the prompt now even if it is not due")
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
