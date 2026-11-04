"""Host setup: plans into temporary homes and workspaces, backups, idempotent re-plans, merges, skills and doctor."""
from __future__ import annotations

import json
import os
import stat
import sys
import tomllib
from pathlib import Path

import pytest

from edgent.application import hosts
from edgent.application.hosts import (
    HOSTS,
    Change,
    apply,
    doctor,
    edgent_command,
    host_paths,
    install_skills,
    next_backup_path,
    plan,
    plan_skills,
    server_entry,
    snippet,
)
from edgent.domain.errors import ConfigurationError

CMD = [sys.executable, "-m", "edgent"]  # an executable that exists, so doctor has nothing to complain about


def expected_entry(workspace, command=CMD) -> dict:
    return {"command": command[0], "args": [*command[1:], "--workspace", str(workspace.root), "mcp"]}


def make_skills(root: Path) -> Path:
    for name, files in {
        "study-workflow": {
            "SKILL.md": "---\nname: study-workflow\ndescription: Route by task.\n---\n# Study workflow\n",
            "references/routing.md": "- study, build, assess\n",
            "scripts/check.py": "print('ok')\n",
            "references/.hidden.md": "skip me\n",
            "scripts/__pycache__/check.cpython-313.pyc": "binary-ish\n",
        },
        "explaining": {"SKILL.md": "---\nname: explaining\ndescription: Find the one idea.\n---\nBody.\n"},
    }.items():
        for rel, text in files.items():
            p = root / name / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text, encoding="utf-8")
    (root / "notes").mkdir()  # no SKILL.md: not a skill
    (root / "notes" / "README.md").write_text("not a skill\n")
    (root / "README.md").write_text("top-level file, ignored\n")
    return root


def make_agents(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "librarian.md").write_text("---\nname: librarian\ndescription: Finds passages.\n---\nFind evidence.\n")
    (root / ".draft.md").write_text("hidden\n")
    return root


def test_edgent_command_and_server_entry(workspace, monkeypatch):
    monkeypatch.setattr(hosts.shutil, "which", lambda name: None)
    assert edgent_command() == [sys.executable, "-m", "edgent"]
    monkeypatch.setattr(hosts.shutil, "which", lambda name: "/usr/local/bin/edgent" if name == "edgent" else None)
    assert edgent_command() == ["/usr/local/bin/edgent"]
    assert server_entry(workspace) == {"command": "/usr/local/bin/edgent", "args": ["--workspace", str(workspace.root), "mcp"]}
    assert server_entry(workspace, "edgent") == {"command": "edgent", "args": ["--workspace", str(workspace.root), "mcp"]}
    assert server_entry(workspace, CMD) == expected_entry(workspace)
    assert server_entry(str(workspace.root), CMD) == expected_entry(workspace) == server_entry(workspace.root, CMD)  # a Path is not duck-typed through Path.root
    with pytest.raises(ConfigurationError):
        server_entry(workspace, [])
    assert json.loads(snippet(workspace, CMD)) == {"mcpServers": {"edgent": expected_entry(workspace)}}
    assert Change("p", "noop", None, "x", "n").to_dict() == {"path": "p", "action": "noop", "note": "n", "before": None, "after": "x"}


def test_host_paths(workspace, home):
    claude = host_paths("claude", workspace, home)
    assert (claude.config, claude.format, claude.skills, claude.agents) == (workspace.root / ".mcp.json", "json", workspace.root / ".claude/skills", workspace.root / ".claude/agents")
    agy = host_paths("agy", workspace, home)
    assert (agy.config, agy.skills, agy.agents) == (home / ".gemini/antigravity-cli/settings.json", workspace.root / ".agents/skills", None)
    codex = host_paths("codex", workspace, home)
    assert (codex.config, codex.format, codex.skills) == (home / ".codex/config.toml", "toml", workspace.root / ".agents/skills")
    cursor = host_paths("cursor", workspace, home)
    assert (cursor.config, cursor.skills, cursor.agents) == (workspace.root / ".cursor/mcp.json", workspace.root / ".cursor/skills", None)
    generic = host_paths("generic", workspace, home)
    assert (generic.config, generic.format, generic.skills, generic.agents) == (None, "none", None, None)
    assert host_paths("agy", workspace).config == home / ".gemini/antigravity-cli/settings.json"  # the default home follows $HOME
    with pytest.raises(ConfigurationError) as ei:
        host_paths("vim", workspace)
    assert "claude, agy, codex, cursor, generic" in str(ei.value)
    with pytest.raises(ConfigurationError):
        plan("vim", workspace)


def test_plan_for_each_host_writes_nothing(workspace, home):
    expected_paths = {
        "claude": workspace.root / ".mcp.json",
        "agy": home / ".gemini" / "antigravity-cli" / "settings.json",
        "codex": home / ".codex" / "config.toml",
        "cursor": workspace.root / ".cursor" / "mcp.json",
    }
    for host, path in expected_paths.items():
        (change,) = plan(host, workspace, home=home, command=CMD)
        assert change.action == "create" and Path(change.path) == path and change.before is None and "edgent" in change.note
        if host == "codex":
            assert tomllib.loads(change.after)["mcp_servers"]["edgent"] == expected_entry(workspace)
            assert change.after.startswith("[mcp_servers.edgent]\n") and change.after.endswith("\n")
        else:
            assert json.loads(change.after)["mcpServers"]["edgent"] == expected_entry(workspace) and change.after.endswith("\n")
        assert not path.exists()
    (generic,) = plan("generic", workspace, home=home, command=CMD)
    assert generic.action == "noop" and generic.path == "" and json.loads(generic.after)["mcpServers"]["edgent"] == expected_entry(workspace)
    assert generic.note.startswith("generic host") and '"mcpServers"' in generic.note
    assert apply(plan("generic", workspace, home=home, command=CMD)) == []


def test_apply_creates_files_and_replan_is_all_noops(workspace, home):
    for host in ("claude", "agy", "codex", "cursor"):
        changes = plan(host, workspace, home=home, command=CMD)
        written = apply(changes)
        assert written == [changes[0].path] and Path(written[0]).read_text() == changes[0].after
        again = plan(host, workspace, home=home, command=CMD)
        assert [c.action for c in again] == ["noop"] and again[0].before == again[0].after == changes[0].after and "already registered" in again[0].note
        assert apply(again) == [] and not list(Path(written[0]).parent.glob("*.bak-*"))
    assert sorted(HOSTS) == sorted(hosts.HOST_SPECS)


def test_merge_preserves_other_servers_and_keys_with_numbered_backups(workspace, home):
    path = workspace.root / ".mcp.json"
    original = json.dumps({"mcpServers": {"other": {"command": "npx", "args": ["-y", "other-server"]}}, "permissions": {"allow": ["Bash"]}}, indent=4) + "\n"
    path.write_text(original)
    (change,) = plan("claude", workspace, home=home, command=CMD)
    assert change.action == "update" and change.before == original and "other servers kept: other" in change.note
    data = json.loads(change.after)
    assert data["mcpServers"]["other"] == {"command": "npx", "args": ["-y", "other-server"]} and data["permissions"] == {"allow": ["Bash"]}
    assert data["mcpServers"]["edgent"] == expected_entry(workspace) and list(data) == ["mcpServers", "permissions"]
    assert apply([change]) == [str(path)]
    assert (workspace.root / ".mcp.json.bak-1").read_text() == original and path.read_text() == change.after

    other_cmd = ["/other/python", "-m", "edgent"]
    (change2,) = plan("claude", workspace, home=home, command=other_cmd)
    assert change2.action == "update" and "command or args changed" in change2.note
    apply([change2])
    assert (workspace.root / ".mcp.json.bak-2").read_text() == change.after and (workspace.root / ".mcp.json.bak-1").read_text() == original
    assert json.loads(path.read_text())["mcpServers"]["edgent"] == expected_entry(workspace, other_cmd)
    apply(plan("claude", workspace, home=home, command=other_cmd), backup=False)
    assert not (workspace.root / ".mcp.json.bak-3").exists()

    # The user's extra keys on the edgent entry survive, and matching command/args mean no rewrite at all.
    with_env = {"mcpServers": {"edgent": {**expected_entry(workspace), "env": {"EDGENT_HOME": "/x"}}, "other": {"command": "npx"}}}
    path.write_text(json.dumps(with_env, indent=2) + "\n")
    (change3,) = plan("claude", workspace, home=home, command=CMD)
    assert change3.action == "noop" and change3.after == path.read_text()
    (change4,) = plan("claude", workspace, home=home, command=other_cmd)
    merged = json.loads(change4.after)["mcpServers"]["edgent"]
    assert change4.action == "update" and merged["env"] == {"EDGENT_HOME": "/x"} and merged["command"] == "/other/python"


def test_backup_numbers_only_increase(tmp_path, monkeypatch):
    f = tmp_path / "settings.json"
    f.write_text("{}")
    assert next_backup_path(f) == tmp_path / "settings.json.bak-1"
    for n in (1, 2, 7):
        (tmp_path / f"settings.json.bak-{n}").write_text("old")
    (tmp_path / "settings.json.bak-x").write_text("not a number")
    (tmp_path / "settings.json.bak-²").write_text("a digit int() rejects")
    assert next_backup_path(f) == tmp_path / "settings.json.bak-8"
    (tmp_path / "settings.json.bak-2").unlink()
    assert next_backup_path(f) == tmp_path / "settings.json.bak-8"  # a gap is never reused
    monkeypatch.setattr(hosts, "BACKUPS_PER_FILE_MAX", 7)
    with pytest.raises(ConfigurationError) as ei:
        next_backup_path(f)
    assert "7 backups" in str(ei.value)


def test_apply_refuses_stale_plans_and_broken_configs(workspace, home):
    path = workspace.root / ".mcp.json"
    changes = plan("claude", workspace, home=home, command=CMD)
    path.write_text('{"mcpServers": {}}\n')
    with pytest.raises(ConfigurationError) as ei:
        apply(changes)
    assert "changed since the plan" in str(ei.value) and path.read_text() == '{"mcpServers": {}}\n'
    for bad, fragment in (("{not json", "not valid JSON"), ("[]", "JSON object"), ('{"mcpServers": 5}', "'mcpServers' must be a JSON object")):
        path.write_text(bad)
        with pytest.raises(ConfigurationError) as ei:
            plan("claude", workspace, home=home, command=CMD)
        assert fragment in str(ei.value) and ".mcp.json" in str(ei.value)
    path.write_text("")
    (change,) = plan("claude", workspace, home=home, command=CMD)
    assert change.action == "update" and json.loads(change.after)["mcpServers"]["edgent"] == expected_entry(workspace)
    path.chmod(0o600)
    apply([change])
    assert stat.S_IMODE(path.stat().st_mode) == 0o600 and (workspace.root / ".mcp.json.bak-1").read_text() == ""
    path.unlink()
    path.mkdir()
    with pytest.raises(ConfigurationError) as ei:
        plan("claude", workspace, home=home, command=CMD)
    assert "is a directory" in str(ei.value)
    toml_path = home / ".codex" / "config.toml"
    toml_path.parent.mkdir(parents=True)
    toml_path.write_text("model = \n")
    with pytest.raises(ConfigurationError) as ei:
        plan("codex", workspace, home=home, command=CMD)
    assert "not valid TOML" in str(ei.value)


def test_codex_toml_append_is_idempotent_and_replaces_only_its_table(workspace, home):
    path = home / ".codex" / "config.toml"
    path.parent.mkdir(parents=True)
    original = 'model = "o3"\n\n[mcp_servers.other]\ncommand = "npx"\nargs = ["-y", "other"]\n\n[mcp_servers.other.env]\nKEY = "v"\n'
    path.write_text(original)
    (change,) = plan("codex", workspace, home=home, command=CMD)
    assert change.action == "update" and change.after.startswith(original + "\n[mcp_servers.edgent]\n") and "append" in change.note
    data = tomllib.loads(change.after)
    assert data["model"] == "o3" and data["mcp_servers"]["other"] == {"command": "npx", "args": ["-y", "other"], "env": {"KEY": "v"}}
    assert data["mcp_servers"]["edgent"] == expected_entry(workspace)
    apply([change])
    (again,) = plan("codex", workspace, home=home, command=CMD)
    assert again.action == "noop" and again.after == path.read_text()

    other_cmd = ["/other/python", "-m", "edgent"]
    (change2,) = plan("codex", workspace, home=home, command=other_cmd)
    assert change2.action == "update" and "replace" in change2.note
    apply([change2])
    text = path.read_text()
    data = tomllib.loads(text)
    assert data["mcp_servers"]["edgent"] == expected_entry(workspace, other_cmd) and data["mcp_servers"]["other"]["env"] == {"KEY": "v"}
    assert text.count("[mcp_servers.edgent]") == 1 and text.startswith('model = "o3"')
    assert (path.parent / "config.toml.bak-1").read_text() == original and (path.parent / "config.toml.bak-2").read_text() == change.after
    assert plan("codex", workspace, home=home, command=other_cmd)[0].action == "noop"

    # Extra keys and sub-tables of an existing edgent table survive a replacement; following tables are untouched.
    path.write_text('[mcp_servers.edgent]\ncommand = "old"\nargs = []\nstartup_timeout_sec = 20\n\n[mcp_servers.edgent.env]\nA = "1"\n\n[other]\nk = 1\n')
    (change3,) = plan("codex", workspace, home=home, command=CMD)
    data = tomllib.loads(change3.after)
    assert data["mcp_servers"]["edgent"] == {**expected_entry(workspace), "startup_timeout_sec": 20, "env": {"A": "1"}} and data["other"] == {"k": 1}
    assert change3.after.count("[mcp_servers.edgent]") == 1 and change3.after.endswith("[other]\nk = 1\n")

    path.write_text('[mcp_servers]\nedgent = { command = "x", args = [] }\n')
    with pytest.raises(ConfigurationError) as ei:
        plan("codex", workspace, home=home, command=CMD)
    assert "inline table" in str(ei.value)
    path.write_text('model = "o3"')  # no trailing newline
    (change4,) = plan("codex", workspace, home=home, command=CMD)
    assert change4.after.startswith('model = "o3"\n\n[mcp_servers.edgent]\n') and tomllib.loads(change4.after)["model"] == "o3"
    path.write_text("")
    (change5,) = plan("codex", workspace, home=home, command=CMD)
    assert change5.action == "update" and change5.after.startswith("[mcp_servers.edgent]\n")
    quoted = ["/opt/with space/python", "-m", "edgent"]
    (change6,) = plan("codex", workspace, home=home, command=quoted)
    assert tomllib.loads(change6.after)["mcp_servers"]["edgent"]["command"] == "/opt/with space/python"


def test_install_skills_and_agents(workspace, tmp_path, monkeypatch):
    skills = make_skills(tmp_path / "builtin_skills")
    agents = make_agents(tmp_path / "builtin_agents")
    changes = install_skills("claude", workspace, source_dir=skills, agents_dir=agents)
    dest = workspace.root / ".claude" / "skills"
    assert {c.action for c in changes} == {"create"}
    assert sorted(Path(c.path).relative_to(workspace.root).as_posix() for c in changes) == [
        ".claude/agents/librarian.md",
        ".claude/skills/explaining/SKILL.md",
        ".claude/skills/study-workflow/SKILL.md",
        ".claude/skills/study-workflow/references/routing.md",
        ".claude/skills/study-workflow/scripts/check.py",
    ]
    assert (dest / "study-workflow" / "scripts" / "check.py").read_text() == "print('ok')\n"
    assert not (dest / "study-workflow" / "references" / ".hidden.md").exists() and not (dest / "study-workflow" / "scripts" / "__pycache__").exists()
    assert not (dest / "notes").exists() and not (dest / "README.md").exists()
    assert (workspace.root / ".claude" / "agents" / "librarian.md").read_text().startswith("---\nname: librarian") and not (workspace.root / ".claude" / "agents" / ".draft.md").exists()
    assert [c.note for c in changes if c.note.startswith("agent")] == ["agent librarian: install"] and "skill study-workflow (SKILL.md): install" in [c.note for c in changes]

    again = install_skills("claude", workspace, source_dir=skills, agents_dir=agents)
    assert [c.action for c in again] == ["noop"] * 5 and all("up to date" in c.note for c in again)
    (skills / "study-workflow" / "SKILL.md").write_text("---\nname: study-workflow\ndescription: Route by task, v2.\n---\n# Study workflow v2\n")
    third = install_skills("claude", workspace, source_dir=skills, agents_dir=agents)
    updated = [c for c in third if c.action != "noop"]
    assert len(updated) == 1 and updated[0].action == "update" and updated[0].path == str(dest / "study-workflow" / "SKILL.md") and "backup" in updated[0].note
    assert (dest / "study-workflow" / "SKILL.md.bak-1").read_text().endswith("# Study workflow\n") and "v2" in (dest / "study-workflow" / "SKILL.md").read_text()

    for host, rel in (("agy", ".agents/skills"), ("codex", ".agents/skills"), ("cursor", ".cursor/skills")):
        changes = install_skills(host, workspace, source_dir=skills, agents_dir=agents)
        assert all(Path(c.path).is_relative_to(workspace.root / rel) for c in changes) and len(changes) == 4  # skills only: these hosts take no agent files
        assert (workspace.root / rel / "explaining" / "SKILL.md").is_file()
    assert install_skills("generic", workspace, source_dir=skills, agents_dir=agents) == []
    assert isinstance(plan_skills("agy", workspace, agents_dir=agents), list)  # the built-in skills directory may be empty
    assert plan_skills("claude", workspace, source_dir=tmp_path / "missing", agents_dir=tmp_path / "missing") == []
    monkeypatch.setattr(hosts, "SKILL_FILES_MAX", 2)
    with pytest.raises(ConfigurationError) as ei:
        plan_skills("claude", workspace, source_dir=skills, agents_dir=agents)
    assert "study-workflow" in str(ei.value) and "2 files" in str(ei.value)


def test_doctor_reports_registration_staleness_and_skills(workspace, home, tmp_path):
    skills = make_skills(tmp_path / "builtin_skills")
    agents = make_agents(tmp_path / "builtin_agents")
    kw = {"home": home, "command": CMD, "source_dir": skills, "agents_dir": agents}
    rep = doctor("claude", workspace, **kw)
    assert rep["registered"] is False and rep["stale"] is False and rep["path"] == str(workspace.root / ".mcp.json")
    assert rep["skills_installed"] == [] and rep["skills_missing"] == ["explaining", "study-workflow"] and rep["skills_stale"] == [] and rep["agents_missing"] == ["librarian"]
    assert any("not registered" in h and "edgent setup --host claude" in h for h in rep["hints"]) and any("missing or outdated" in h for h in rep["hints"])
    assert not (workspace.root / ".mcp.json").exists()  # doctor never writes

    apply(plan("claude", workspace, home=home, command=CMD))
    install_skills("claude", workspace, source_dir=skills, agents_dir=agents)
    rep = doctor("claude", workspace, **kw)
    assert rep["registered"] is True and rep["stale"] is False and rep["hints"] == []
    assert rep["skills_installed"] == ["explaining", "study-workflow"] and rep["skills_missing"] == [] and rep["agents_missing"] == []

    stale = doctor("claude", workspace, home=home, command=["/other/python", "-m", "edgent"], source_dir=skills, agents_dir=agents)
    assert stale["registered"] is True and stale["stale"] is True and any("differ" in h and "setup" in h for h in stale["hints"])

    (workspace.root / ".claude" / "skills" / "study-workflow" / "SKILL.md").write_text("edited by hand\n")
    rep = doctor("claude", workspace, **kw)
    assert rep["skills_stale"] == ["study-workflow"] and rep["skills_installed"] == ["explaining"] and any("outdated" in h for h in rep["hints"])

    ghost = ["/nonexistent/edgent"]
    apply(plan("claude", workspace, home=home, command=ghost))
    rep = doctor("claude", workspace, home=home, command=ghost, source_dir=skills, agents_dir=agents)
    assert rep["registered"] and not rep["stale"] and any("not on PATH" in h for h in rep["hints"])

    (workspace.root / ".mcp.json").write_text("{")
    rep = doctor("claude", workspace, **kw)
    assert rep["registered"] is False and any("not valid JSON" in h for h in rep["hints"])

    generic = doctor("generic", workspace, **kw)
    assert generic["registered"] is False and generic["path"] == "" and generic["skills_missing"] == [] and "generic host" in generic["hints"][0]

    codex = doctor("codex", workspace, **kw)
    assert codex["registered"] is False and codex["path"] == str(home / ".codex" / "config.toml") and codex["agents_missing"] == []
    apply(plan("codex", workspace, home=home, command=CMD))
    codex = doctor("codex", workspace, **kw)
    assert codex["registered"] is True and codex["stale"] is False and codex["skills_missing"] == ["explaining", "study-workflow"]
    assert doctor("agy", workspace, **kw)["path"] == str(home / ".gemini" / "antigravity-cli" / "settings.json")
    assert os.environ["HOME"] == str(home) and doctor("agy", workspace, command=CMD, source_dir=skills)["path"] == str(home / ".gemini" / "antigravity-cli" / "settings.json")


def test_misplaced_or_unwritable_configs_are_typed_errors_and_doctor_hints(workspace, home, tmp_path, monkeypatch):
    skills = make_skills(tmp_path / "builtin_skills")
    (workspace.root / ".mcp.json").mkdir()
    rep = doctor("claude", workspace, home=home, command=CMD, source_dir=skills, agents_dir=tmp_path / "none")
    assert rep["registered"] is False and any("is a directory" in h for h in rep["hints"])  # a hint, never an exception
    (home / ".codex").write_text("not a directory\n")
    with pytest.raises(ConfigurationError) as ei:
        plan("codex", workspace, home=home, command=CMD)
    assert "parent of this path is a file" in str(ei.value) and "config.toml" in str(ei.value)
    assert any("parent of this path is a file" in h for h in doctor("codex", workspace, home=home, command=CMD, source_dir=skills)["hints"])

    changes = plan("cursor", workspace, home=home, command=CMD)
    monkeypatch.setattr(hosts, "atomic_write_text", lambda path, text: (_ for _ in ()).throw(PermissionError(13, "Permission denied")))
    with pytest.raises(ConfigurationError) as ei:
        apply(changes)
    assert "cannot write" in str(ei.value) and "Permission denied" in str(ei.value) and "mcp.json" in str(ei.value)
    assert not (workspace.root / ".cursor" / "mcp.json").exists()


def test_apply_writes_through_a_symlinked_config(workspace, home):
    real = home / "dotfiles" / "claude-mcp.json"
    real.parent.mkdir()
    original = json.dumps({"mcpServers": {"other": {"command": "npx"}}}, indent=2) + "\n"
    real.write_text(original)
    link = workspace.root / ".mcp.json"
    link.symlink_to(real)
    (change,) = plan("claude", workspace, home=home, command=CMD)
    assert change.action == "update" and change.before == original
    assert apply([change]) == [str(link)]
    assert link.is_symlink() and link.resolve() == real.resolve()  # the dotfiles link survives
    assert json.loads(real.read_text())["mcpServers"]["edgent"] == expected_entry(workspace)
    assert (real.parent / "claude-mcp.json.bak-1").read_text() == original and not list(workspace.root.glob("*.bak-*"))
    assert plan("claude", workspace, home=home, command=CMD)[0].action == "noop"


def test_non_utf8_skill_files_are_skipped_not_corrupted(workspace, tmp_path):
    skills = make_skills(tmp_path / "builtin_skills")
    binary = skills / "study-workflow" / "references" / "figure.png"
    binary.write_bytes(b"\x89PNG\r\n\x1a\n\x00\xff\xfe")
    (skills / "explaining" / "SKILL.md").write_bytes("---\nname: explaining\n---\nCaf\u00e9 \u2014 UTF-8 stays intact.\n".encode())
    changes = plan_skills("agy", workspace, source_dir=skills)
    skipped = [c for c in changes if "skipped" in c.note]
    assert len(skipped) == 1 and skipped[0].action == "noop" and skipped[0].path.endswith("references/figure.png") and "not UTF-8" in skipped[0].note
    install_skills("agy", workspace, source_dir=skills)
    dest = workspace.root / ".agents" / "skills"
    assert not (dest / "study-workflow" / "references" / "figure.png").exists()
    assert (dest / "explaining" / "SKILL.md").read_text(encoding="utf-8") == "---\nname: explaining\n---\nCaf\u00e9 \u2014 UTF-8 stays intact.\n"
    assert (dest / "study-workflow" / "references" / "routing.md").is_file()
    rep = doctor("agy", workspace, home=tmp_path / "h", command=CMD, source_dir=skills)
    assert rep["skills_installed"] == ["explaining", "study-workflow"] and rep["skills_stale"] == [] and not any("outdated" in h for h in rep["hints"])


def test_setup_never_writes_through_a_symlink_under_the_host_trees(workspace, tmp_path):
    """A link the model planted under .claude/skills or .claude/agents must not turn setup into a write anywhere else."""
    skills = make_skills(tmp_path / "builtin_skills")
    agents = make_agents(tmp_path / "builtin_agents")
    victim = tmp_path / "victim" / "authorized_keys"
    victim.parent.mkdir()
    victim.write_text("ssh-ed25519 AAAA real-key\n")
    link = workspace.root / ".claude" / "skills" / "study-workflow" / "SKILL.md"
    link.parent.mkdir(parents=True)
    link.symlink_to(victim)
    with pytest.raises(ConfigurationError) as ei:
        plan_skills("claude", workspace, source_dir=skills, agents_dir=agents)
    assert "symbolic link" in str(ei.value) and str(link) in str(ei.value) and "Remove the link" in str(ei.value)
    assert victim.read_text() == "ssh-ed25519 AAAA real-key\n" and not list(victim.parent.glob("*.bak-*"))
    rep = doctor("claude", workspace, home=tmp_path / "h", command=CMD, source_dir=skills, agents_dir=agents)
    assert any("symbolic link" in h for h in rep["hints"])  # doctor reports it instead of crashing
    link.unlink()

    # A linked directory is refused too, for the agent tree.
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (workspace.root / ".claude" / "agents").symlink_to(elsewhere, target_is_directory=True)
    with pytest.raises(ConfigurationError, match="symbolic link"):
        plan_skills("claude", workspace, source_dir=skills, agents_dir=agents)
    assert not (elsewhere / "librarian.md").exists()
    (workspace.root / ".claude" / "agents").unlink()

    # A link planted between the plan and the apply is caught by apply; the config dotfile link keeps working.
    changes = plan_skills("claude", workspace, source_dir=skills, agents_dir=agents)
    assert changes and all(c.follow_symlinks is False for c in changes)
    agent_link = workspace.root / ".claude" / "agents" / "librarian.md"
    agent_link.parent.mkdir(parents=True)
    agent_link.symlink_to(victim)
    with pytest.raises(ConfigurationError, match="symbolic link"):
        apply(changes)
    assert victim.read_text() == "ssh-ed25519 AAAA real-key\n" and not list(victim.parent.glob("*.bak-*"))
    agent_link.unlink()
    apply(changes)
    assert agent_link.is_file() and not agent_link.is_symlink()
    assert Change("x", "create", None, "", "").follow_symlinks is True  # the config change keeps the dotfiles behaviour


def test_claude_agents_are_installed_with_the_host_tool_names(workspace, tmp_path):
    from edgent.application.hosts import CLAUDE_TOOL_NAMES, MCP_TOOL_PREFIX, claude_tool_names, host_agent_text
    from edgent.application.subagents import AgentDefinition

    agents = tmp_path / "agents"
    agents.mkdir()
    body = "You are the author.\n\nUse `read_file`, then `check_cite`; a ```tool fence stays as is.\n"
    source = (
        "---\nname: author\ndescription: Drafts one unit: the brief, the sources, the checks.\n"
        "tools: read_file, write_file, grep, sources_cite_check, check_cite, run_python, todo_write, todo_read, session_recall, remember, skill, ask_user, agent\n"
        "effort: high\nskills: [book-authoring, explaining]\nmaxTurns: 60\nbackground: false\n---\n" + body
    )
    (agents / "author.md").write_text(source)
    (agents / "all.md").write_text("---\nname: all\ndescription: Everything.\ntools: *\neffort: low\n---\nBody.\n")
    (agents / "listed.md").write_text("---\nname: listed\ndescription: Multi-line list.\ntools:\n  - read_file\n  - 'film_look'\ndisallowedTools: bash\n---\nBody.\n")
    (agents / "bare.md").write_text("No front matter at all.\n")
    skills = make_skills(tmp_path / "builtin_skills")
    changes = install_skills("claude", workspace, source_dir=skills, agents_dir=agents)
    installed = (workspace.root / ".claude" / "agents" / "author.md").read_text()
    expected_tools = (
        "tools: Read, mcp__edgent__read_file, Write, mcp__edgent__write_file, Grep, mcp__edgent__grep, mcp__edgent__sources_cite_check, "
        "mcp__edgent__check_cite, mcp__edgent__run_python, TodoWrite, Skill, mcp__edgent__skill, AskUserQuestion, Task"
    )
    assert installed == (
        "---\nname: author\ndescription: Drafts one unit: the brief, the sources, the checks.\n"
        f"{expected_tools}\nskills: [book-authoring, explaining]\nmaxTurns: 60\nbackground: false\n---\n" + body
    )
    assert "session_recall" not in installed.split("---")[1] and "effort" not in installed and "mcp__edgent__todo" not in installed
    assert (workspace.root / ".claude" / "agents" / "all.md").read_text() == "---\nname: all\ndescription: Everything.\ntools: *\n---\nBody.\n"
    assert (workspace.root / ".claude" / "agents" / "listed.md").read_text() == (
        "---\nname: listed\ndescription: Multi-line list.\ntools: Read, mcp__edgent__read_file, mcp__edgent__film_look\n"
        "disallowedTools: Bash, mcp__edgent__bash\n---\nBody.\n"
    )
    assert (workspace.root / ".claude" / "agents" / "bare.md").read_text() == "No front matter at all.\n"
    # The rewritten file still parses as an agent definition, and a second install is a noop (the plan compares rewritten text).
    parsed = AgentDefinition.from_markdown(installed, path=workspace.root / ".claude" / "agents" / "author.md")
    assert parsed.name == "author" and parsed.tools and parsed.tools[0] == "Read" and parsed.system_prompt == body.strip()
    assert all(c.action == "noop" for c in install_skills("claude", workspace, source_dir=skills, agents_dir=agents))
    assert [c.note for c in changes if c.note.startswith("agent a")] == ["agent all: install", "agent author: install"]
    # Only Claude Code reads agent files; other hosts' text is untouched, as is a file without a closing fence.
    assert host_agent_text("agy", source) == source and host_agent_text("codex", source) == source
    assert host_agent_text("claude", "---\nname: x\ntools: read_file\nno closing fence\n") == "---\nname: x\ntools: read_file\nno closing fence\n"
    assert claude_tool_names(["read_file", "read_file", "list_files", "glob"]) == ["Read", f"{MCP_TOOL_PREFIX}read_file", "Glob", f"{MCP_TOOL_PREFIX}list_files", f"{MCP_TOOL_PREFIX}glob"]
    assert all(name.startswith("mcp__edgent__") or "_" not in name for names in CLAUDE_TOOL_NAMES.values() for name in names)


def test_builtin_agents_rewrite_to_tools_the_mcp_server_serves(workspace):
    """Every built-in agent, installed for Claude Code, names only host tools or MCP names of tools the registry really has."""
    from edgent.application.hosts import MCP_TOOL_PREFIX, host_agent_text
    from edgent.application.subagents import AgentDefinition, builtin_agents_dir
    from edgent.tools.checks.registry import build_checks_registry
    from edgent.tools.core import build_core_registry
    from edgent.tools.edu.assess import build_assess_registry
    from edgent.tools.edu.book import build_book_registry
    from edgent.tools.edu.learner import build_learner_registry
    from edgent.tools.edu.media import build_media_registry
    from edgent.tools.edu.sources import build_sources_registry
    from edgent.tools.mcp_server import HOST_CONTEXT_TOOLS
    from edgent.tools.registry import ToolRegistry

    reg = ToolRegistry()
    for build in (build_core_registry, build_sources_registry, build_book_registry, build_checks_registry, build_media_registry, build_assess_registry, build_learner_registry):
        build(reg)
    served = set(reg.names()) - set(HOST_CONTEXT_TOOLS)
    files = sorted(Path(builtin_agents_dir()).glob("*.md"))
    assert len(files) >= 10
    for f in files:
        original = AgentDefinition.from_markdown(f.read_text(encoding="utf-8"), path=f)
        rewritten = AgentDefinition.from_markdown(host_agent_text("claude", f.read_text(encoding="utf-8")), path=f)
        assert rewritten.system_prompt == original.system_prompt and rewritten.skills == original.skills and rewritten.max_turns == original.max_turns
        assert original.tools and rewritten.tools, f.name
        for tool in rewritten.tools:
            assert tool.startswith(MCP_TOOL_PREFIX) or tool[0].isupper(), (f.name, tool)  # a Claude Code name, never a bare Edgent id
            if tool.startswith(MCP_TOOL_PREFIX):
                assert tool[len(MCP_TOOL_PREFIX) :] in served, (f.name, tool)
