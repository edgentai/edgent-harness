"""The golden scenarios run through the real loop, the real tools and the gate."""
from __future__ import annotations

import shutil
from pathlib import Path

from edgent.evals.gate import load_scenarios
from edgent.evals.runner import run_scenarios
from edgent.ui.cli import main

GOLDEN = Path(__file__).resolve().parents[1] / "evals" / "golden"


def test_golden_scenarios_pass(tmp_path, home):
    scenarios = load_scenarios(GOLDEN)
    results = run_scenarios(scenarios, tmp_path)
    failed = [(r.scenario_id, r.failed_checks()) for r in results if not r.passed]
    assert len(results) >= 6 and not failed, failed
    assert all(r.tool_calls >= 1 for r in results)


def test_eval_command_records_results(tmp_path, home, capsys):
    golden = tmp_path / "evals" / "golden"
    shutil.copytree(GOLDEN, golden)
    assert main(["eval", str(golden), "--label", "unit-test", "--only", "tutor-cite-passage", "tutor-beyond-sources"]) == 0
    out = capsys.readouterr().out
    assert out.count("PASS") == 2 and "unit-test" in out
    assert "unit-test" in (tmp_path / "evals" / "results.md").read_text(encoding="utf-8")
