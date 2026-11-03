"""The command line, end to end with the mock provider and a private home."""
from __future__ import annotations

import json

from edgent.ui.cli import main


def _ws(tmp_path):
    ws = tmp_path / "ws"
    assert main(["init", str(ws), "--no-git"]) == 0
    assert (ws / "STUDY.md").exists()
    return ws


def test_init_run_sessions_status_doctor_skills(tmp_path, home, capsys):
    ws = _ws(tmp_path)
    capsys.readouterr()
    assert main(["--workspace", str(ws), "--provider", "mock", "run", "hello there"]) == 0
    out = capsys.readouterr()
    assert "[mock]" in out.out and "session:" in out.err
    assert main(["run", "--workspace", str(ws), "--provider", "mock", "--json", "second"]) == 0  # global options after the command
    assert json.loads(capsys.readouterr().out)["stop_reason"] == "done"
    assert main(["--workspace", str(ws), "--provider", "mock", "sessions"]) == 0
    assert len(capsys.readouterr().out.strip().splitlines()) == 2
    assert main(["--workspace", str(ws), "--provider", "mock", "sessions", "--kind", "build"]) == 0
    assert capsys.readouterr().out.strip() == ""
    assert main(["--workspace", str(ws), "--provider", "mock", "status"]) == 0
    assert "unknowns_open" in capsys.readouterr().out
    assert main(["--workspace", str(ws), "--provider", "mock", "doctor", "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["provider"]["kind"] == "mock" and d["workspace"]["git"] is False
    assert main(["--workspace", str(ws), "--provider", "mock", "doctor"]) == 0
    assert "provider: mock" in capsys.readouterr().out
    assert main(["--workspace", str(ws), "--provider", "mock", "skills"]) == 0
    assert "explaining" in capsys.readouterr().out
    assert main(["--workspace", str(ws), "--provider", "mock", "skills", "agents"]) == 0
    assert "librarian" in capsys.readouterr().out
    assert main(["--workspace", str(ws), "--provider", "mock", "outline"]) == 0
    assert main(["--workspace", str(ws), "--provider", "mock", "walkthrough"]) == 0


def test_sources_add_list_and_failures(tmp_path, home, capsys):
    ws = _ws(tmp_path)
    src = tmp_path / "notes.md"
    src.write_text("# Notes\n\nThe damping factor must be below one.\n", encoding="utf-8")
    capsys.readouterr()
    assert main(["--workspace", str(ws), "--provider", "mock", "sources", "add", str(src)]) == 0
    assert "added notes" in capsys.readouterr().out
    assert main(["--workspace", str(ws), "--provider", "mock", "sources", "list"]) == 0
    assert "notes" in capsys.readouterr().out
    assert main(["--workspace", str(ws), "--provider", "mock", "sources", "add", str(tmp_path / "missing.pdf")]) == 1
    assert "failed" in capsys.readouterr().err


def test_no_workspace_and_no_provider_are_clear_errors(tmp_path, home, capsys, monkeypatch):
    assert main(["--workspace", str(tmp_path / "none"), "run", "x"]) == 2
    assert "edgent init" in capsys.readouterr().err
    ws = _ws(tmp_path)
    capsys.readouterr()
    monkeypatch.delenv("EDGENT_PROVIDER", raising=False)  # nothing configured at all: no key, no CLI on the PATH
    monkeypatch.setenv("PATH", str(tmp_path / "empty-bin"))
    for key in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL"):
        monkeypatch.delenv(key, raising=False)
    assert main(["--workspace", str(ws), "run", "x"]) == 0  # the offline mock answers and a warning says why
    err = capsys.readouterr().err
    assert "no provider configured" in err and "offline mock" in err


def test_setup_dry_run_plans_host_files(tmp_path, home, capsys):
    ws = _ws(tmp_path)
    capsys.readouterr()
    assert main(["--workspace", str(ws), "--provider", "mock", "setup", "--host", "claude", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert ".mcp.json" in out and "register edgent" in out and "agents/librarian.md" in out
    assert not (ws / ".mcp.json").exists()  # a dry run writes nothing


def test_build_without_an_outline_asks_for_a_goal(tmp_path, home, capsys):
    ws = _ws(tmp_path)
    capsys.readouterr()
    assert main(["--workspace", str(ws), "--provider", "mock", "build"]) == 2
    assert "goal" in capsys.readouterr().err
