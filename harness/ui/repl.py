"""Interactive streaming terminal REPL (Claude-Code-style UX, stdlib only).

Features:
- Real-time streamed text, optional thinking, and tool start/end status lines.
- Interactive permission prompts (``[y/N/a(lways)]``) in ``ask`` / ``accept_edits`` mode.
- Slash commands: ``/help``, ``/mode``, ``/session``, ``/sessions``, ``/skills``,
  ``/subagents``, ``/tools``, ``/tasks``, ``/undo``, ``/checkpoints``, ``/compact``,
  ``/recall``, ``/approve``, ``/reject``, ``/thinking``, ``/quit``.
- Ctrl-C cancels the active turn via ``cancel_event`` without killing the REPL.
"""
from __future__ import annotations

import json
import signal
import sys
import threading
from typing import Any, Optional, TextIO

from ..core.client import StreamEvent
from ..core.utils import truncate
from ..session.loop import AgentLoop, TurnResult
from ..session.policy import PermissionMode

HELP_TEXT = """Commands:
  /help                         Show this help
  /mode [ask|accept_edits|auto|plan]  Show or switch permission mode
  /session [id]                 Show or switch active session id
  /sessions                     List saved sessions
  /skills                       List available and loaded skills
  /subagents                    List available subagent roles
  /tools                        List registered tools
  /tasks                        List background shell tasks
  /undo [N]                     Rewind workspace edits from the last N turns
  /checkpoints                  List checkpoints for the active session
  /compact                      Compact older transcript turns into rolling notes
  /recall <query>               Search earlier session turns verbatim
  /approve [feedback]           Approve a pending request_review and resume
  /reject [feedback]            Request changes on a pending review and resume
  /thinking [on|off]            Toggle printing model reasoning deltas
  /quit | /exit                 Exit the REPL"""


class TerminalREPL:
    """Streaming interactive terminal shell around an ``AgentLoop``."""

    def __init__(
        self,
        loop: AgentLoop,
        session_id: str = "default",
        *,
        stdin: Optional[TextIO] = None,
        stdout: Optional[TextIO] = None,
        stderr: Optional[TextIO] = None,
        show_thinking: bool = False,
    ) -> None:
        self.loop = loop
        self.session_id = session_id
        self.stdin = stdin or sys.stdin
        self.stdout = stdout or sys.stdout
        self.stderr = stderr or sys.stderr
        self.show_thinking = show_thinking
        self._running = True
        self._streamed_any_text = False
        if self.loop.approve is None:
            self.loop.approve = self._prompt_approve

    def _write(self, text: str, *, end: str = "\n", stream: Optional[TextIO] = None) -> None:
        out = stream or self.stdout
        out.write(text + end)
        out.flush()

    def _prompt_approve(self, tool_name: str, args: dict[str, Any]) -> bool:
        preview = truncate(json.dumps(args, ensure_ascii=False, default=str), 140)
        self._write(f"\n[approve] {tool_name}({preview})? [y/N/a(lways)]: ", end="", stream=self.stderr)
        line = self.stdin.readline()
        if not line:
            return False
        ans = line.strip().lower()
        if ans in ("a", "always"):
            if tool_name not in self.loop.policy.allow:
                self.loop.policy.allow.append(tool_name)
            return True
        return ans in ("y", "yes")

    def _on_event(self, ev: StreamEvent) -> None:
        if ev.kind == "thinking_delta":
            if self.show_thinking and ev.text:
                self._write(ev.text, end="", stream=self.stderr)
        elif ev.kind == "text_delta":
            if ev.text:
                self._streamed_any_text = True
                self._write(ev.text, end="")
        elif ev.kind == "tool_start":
            if self._streamed_any_text:
                self._write("")
                self._streamed_any_text = False
            name = ev.data.get("name", "")
            args = truncate(json.dumps(ev.data.get("arguments") or {}, ensure_ascii=False, default=str), 110)
            self._write(f"  > {name}({args})", stream=self.stderr)
        elif ev.kind == "tool_end":
            name = ev.data.get("name", "")
            status = "ok" if ev.data.get("ok") else "err"
            dur = ev.data.get("duration_s", 0.0)
            self._write(f"  < {name} [{status} {dur:.2f}s]", stream=self.stderr)
        elif ev.kind == "status":
            if ev.text:
                self._write(f"  [{ev.text}]", stream=self.stderr)
        elif ev.kind == "error":
            if ev.text:
                self._write(f"  [error] {ev.text}", stream=self.stderr)

    def _render_result(self, res: TurnResult) -> None:
        if self._streamed_any_text:
            self._write("")
            self._streamed_any_text = False
        elif res.reply:
            self._write(res.reply)
        if res.status == "awaiting_review" and res.review_request:
            req = res.review_request
            self._write(f"\n[awaiting review] {req.get('title', '')}", stream=self.stderr)
            if req.get("summary"):
                self._write(f"  {req['summary']}", stream=self.stderr)
            if req.get("artifact_paths"):
                self._write(f"  artifacts: {', '.join(req['artifact_paths'])}", stream=self.stderr)
            self._write("  use `/approve [note]` or `/reject <feedback>` to continue", stream=self.stderr)
        meta = f"[{res.status} | turns={res.turns_used} tools={res.tool_calls_used} tokens={res.usage.total_tokens}"
        if res.checkpoint_id:
            meta += f" cp={res.checkpoint_id}"
        meta += "]"
        self._write(meta, stream=self.stderr)

    def _run_cancellable(self, fn) -> TurnResult:
        cancel = threading.Event()
        holder: dict[str, Any] = {}

        def _worker() -> None:
            try:
                holder["res"] = fn(cancel)
            except Exception as e:  # noqa: BLE001
                holder["err"] = e

        is_main = threading.current_thread() is threading.main_thread()
        prev_handler = None
        if is_main:
            def _sigint(_sig, _frame) -> None:
                cancel.set()
                self._write("\n[interrupt requested — stopping turn...]", stream=self.stderr)

            prev_handler = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, _sigint)
        try:
            t = threading.Thread(target=_worker, daemon=True)
            t.start()
            while t.is_alive():
                t.join(timeout=0.1)
        finally:
            if is_main and prev_handler is not None:
                signal.signal(signal.SIGINT, prev_handler)

        if "err" in holder:
            raise holder["err"]
        return holder["res"]

    def handle_command(self, line: str) -> str:
        """Execute a slash command and return its text output."""
        parts = line.strip().split(None, 1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        if cmd in ("/quit", "/exit"):
            self._running = False
            return "Goodbye."
        if cmd == "/help":
            return HELP_TEXT
        if cmd == "/mode":
            if not arg:
                return f"mode: {self.loop.policy.mode.value} (allow={self.loop.policy.allow}, deny={self.loop.policy.deny})"
            try:
                self.loop.policy.mode = PermissionMode(arg)
                return f"mode switched to: {self.loop.policy.mode.value}"
            except ValueError:
                return f"invalid mode '{arg}'; expected one of: ask, accept_edits, auto, plan"
        if cmd == "/session":
            if arg:
                self.session_id = arg
                return f"switched to session: {self.session_id}"
            st = self.loop.sessions.load_state(self.session_id)
            return f"session: {st.session_id} (status={st.status}, turns={st.turn_count})"
        if cmd == "/sessions":
            items = self.loop.sessions.list_sessions()
            if not items:
                return "no sessions yet"
            return "\n".join(f"  {s['session_id']:20s} status={s.get('status'):15s} turns={s.get('turn_count', 0)}" for s in items)
        if cmd == "/skills":
            st = self.loop.sessions.load_state(self.session_id)
            avail = self.loop.skills.names()
            return f"available: {avail or '(none)'}\nloaded:    {st.loaded_skills or '(none)'}"
        if cmd == "/subagents":
            return "\n".join(f"  {s.name:14s} — {s.description}" for s in self.loop.subagents.all())
        if cmd == "/tools":
            return ", ".join(self.loop.registry.names())
        if cmd == "/tasks":
            tasks = self.loop.list_tasks()
            if not tasks:
                return "no background tasks"
            return "\n".join(
                f"  {t['task_id']} running={t['running']} rc={t['returncode']} ({t['elapsed_s']}s) {truncate(t['command'], 60)}"
                for t in tasks
            )
        if cmd == "/undo":
            steps = int(arg) if arg.isdigit() else 1
            res = self.loop.undo(self.session_id, steps=steps)
            return json.dumps(res, indent=2)
        if cmd == "/checkpoints":
            cps = self.loop.checkpoints.list_checkpoints(self.session_id)
            if not cps:
                return "no checkpoints recorded for this session"
            return "\n".join(f"  {c['id']} ({len(c.get('files') or [])} files) {c.get('summary', '')}" for c in cps)
        if cmd == "/compact":
            summary = self.loop.compact(self.session_id)
            return "compacted older turns into notes.md" if summary else "nothing to compact yet"
        if cmd == "/recall":
            if not arg:
                return "usage: /recall <query>"
            hits = self.loop.sessions.recall(self.session_id, arg)
            if not hits:
                return "no matches found"
            return "\n".join(f"  [#{h['seq']} {h['role']}] {truncate(h['text'], 160)}" for h in hits)
        if cmd in ("/approve", "/reject"):
            approved = cmd == "/approve"
            self._streamed_any_text = False
            res = self._run_cancellable(
                lambda c: self.loop.resume_review(self.session_id, approved=approved, feedback=arg, on_event=self._on_event, cancel_event=c)
            )
            self._render_result(res)
            return ""
        if cmd == "/thinking":
            if arg.lower() in ("on", "true", "1"):
                self.show_thinking = True
            elif arg.lower() in ("off", "false", "0"):
                self.show_thinking = False
            else:
                self.show_thinking = not self.show_thinking
            return f"thinking output: {'on' if self.show_thinking else 'off'}"
        return f"unknown command '{cmd}'; type /help for commands"

    def handle_line(self, line: str) -> Optional[TurnResult]:
        """Process one input line (slash command or user message)."""
        text = line.strip()
        if not text:
            return None
        if text.startswith("/"):
            out = self.handle_command(text)
            if out:
                self._write(out)
            return None
        self._streamed_any_text = False
        res = self._run_cancellable(
            lambda c: self.loop.send(self.session_id, text, on_event=self._on_event, cancel_event=c)
        )
        self._render_result(res)
        return res

    def run(self) -> int:
        self._write(
            f"Edgent REPL | workspace={self.loop.workspace.path} | session={self.session_id} | mode={self.loop.policy.mode.value}",
            stream=self.stderr,
        )
        self._write("Type /help for commands, Ctrl-C to interrupt a turn, /quit to exit.", stream=self.stderr)
        while self._running:
            try:
                self._write(f"\n[{self.session_id}:{self.loop.policy.mode.value}]> ", end="", stream=self.stderr)
                raw = self.stdin.readline()
            except KeyboardInterrupt:
                self._write("\n(use /quit to exit)", stream=self.stderr)
                continue
            if not raw:
                break
            self.handle_line(raw)
        return 0


def run_repl(
    loop: AgentLoop,
    session_id: str = "default",
    *,
    show_thinking: bool = False,
) -> int:
    return TerminalREPL(loop, session_id=session_id, show_thinking=show_thinking).run()
