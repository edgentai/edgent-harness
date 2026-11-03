"""Shared fixtures: a fresh workspace, a mock provider, a registry with a context."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from edgent.adapters.providers.mock import MockProvider  # noqa: E402
from edgent.adapters.toolchain import Toolchain  # noqa: E402
from edgent.domain.evidence import EvidenceStore  # noqa: E402
from edgent.domain.workspace import Workspace  # noqa: E402
from edgent.tools.registry import AllowAll, IdempotencyStore, ToolContext, ToolRegistry  # noqa: E402


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setenv("EDGENT_HOME", str(h))
    monkeypatch.setenv("HOME", str(h))
    monkeypatch.setenv("EDGENT_PROVIDER", "mock")  # never let a test reach a real key or an installed agy/claude CLI
    return h


@pytest.fixture
def workspace(tmp_path: Path, home: Path) -> Workspace:
    return Workspace.init(tmp_path / "ws", name="Test Workspace", git=True)


@pytest.fixture
def provider() -> MockProvider:
    return MockProvider()


@pytest.fixture
def registry(workspace: Workspace) -> ToolRegistry:
    return ToolRegistry(idempotency=IdempotencyStore(workspace.path("state") / "state.db"))


@pytest.fixture
def ctx(workspace: Workspace, provider: MockProvider) -> ToolContext:
    return ToolContext(
        workspace=workspace,
        session_id="s1",
        provider=provider,
        evidence=EvidenceStore(workspace.path("evidence")),
        toolchain=Toolchain.detect(),
        permissions=AllowAll(),
    )


def skip_unless(toolchain_name: str):
    tc = Toolchain.detect()
    return pytest.mark.skipif(not tc.has(toolchain_name), reason=f"{toolchain_name} not installed; this check is unverified here")
