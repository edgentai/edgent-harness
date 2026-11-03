from __future__ import annotations

import time

import pytest

from edgent.domain.errors import ToolFailure
from edgent.tools.registry import (
    AskDangerous,
    I,
    IdempotencyStore,
    ReadOnly,
    S,
    ToolRegistry,
    ToolResult,
    ToolSpec,
    compact_schema,
    obj,
    validate_args,
)


def make_registry(workspace, **kw):
    reg = ToolRegistry(idempotency=IdempotencyStore(workspace.path("state") / "idem.db"), **kw)

    @reg.tool("echo", "Echo arguments back.", obj({"text": S, "n": {"type": "integer", "minimum": 0}}, ["text"]))
    def echo(ctx, a):
        return {"text": a["text"], "n": a.get("n", 0)}

    @reg.tool("big", "Return a big output.", obj({}))
    def big(ctx, a):
        return {"data": "x" * 20_000}

    @reg.tool("slow", "Sleep.", obj({}), timeout_s=0.2)
    def slow(ctx, a):
        time.sleep(1.0)
        return {}

    @reg.tool("fails", "Raise a typed failure.", obj({}))
    def fails(ctx, a):
        raise ToolFailure("unavailable", "no ffmpeg", "unavailable", remediation="install ffmpeg", partial={"done": 3})

    @reg.tool("boom", "Raise an unexpected exception.", obj({}))
    def boom(ctx, a):
        raise ValueError("kaboom")

    counter = {"n": 0}

    @reg.tool("write_thing", "Side effect.", obj({"k": S}, ["k"]), permission="write", side_effects=True, idempotent=False)
    def write_thing(ctx, a):
        counter["n"] += 1
        return {"written": a["k"], "count": counter["n"]}

    @reg.tool("delete_all", "Dangerous.", obj({}), permission="delete")
    def delete_all(ctx, a):
        return {"deleted": True}

    @reg.tool("attach", "Attach an image.", obj({}), attaches_media=True)
    def attach(ctx, a):
        p = ctx.workspace.write_text("media/x.png", "notreallypng")
        return {"_attach": [{"type": "image", "path": str(p)}], "ok": True}

    return reg, counter


def test_validation_and_contract(workspace, ctx):
    reg, _ = make_registry(workspace)
    r = reg.call("echo", {"text": "hi", "n": 2}, ctx)
    assert r.ok and r.output == {"text": "hi", "n": 2}
    r = reg.call("echo", {"n": -1}, ctx)
    assert not r.ok and r.error.category.value == "invalid_input" and "required" in r.error.message and r.error.recovery["kind"] == "inspect_input_contract"
    r = reg.call("nope", {}, ctx)
    assert r.error.code == "not_found" and "echo" in r.error.message
    r = reg.call("echo", "notadict", ctx)
    assert r.error.code == "invalid_input"
    assert validate_args(obj({"a": {"anyOf": [S, I]}}), {"a": 1.5}) and not validate_args(obj({"a": {"anyOf": [S, I]}}), {"a": 1})


def test_permissions(workspace, ctx):
    reg, _ = make_registry(workspace)
    ctx.permissions = ReadOnly()
    assert reg.call("write_thing", {"k": "a"}, ctx).error.code == "permission_denied"
    ctx.permissions = AskDangerous()
    ctx.ask_permission = None
    r = reg.call("delete_all", {}, ctx)
    assert r.error.code == "permission_denied" and "no way to ask" in r.error.message
    ctx.ask_permission = lambda spec, args: True
    assert reg.call("delete_all", {}, ctx).ok
    ctx.ask_permission = lambda spec, args: False
    assert "declined" in reg.call("delete_all", {}, ctx).error.message
    assert reg.call("echo", {"text": "x"}, ctx, allowlist={"big"}).error.code == "permission_denied"


def test_idempotency_and_replay(workspace, ctx):
    reg, counter = make_registry(workspace)
    r1 = reg.call("write_thing", {"k": "a"}, ctx)
    r2 = reg.call("write_thing", {"k": "a"}, ctx)
    assert r1.output["count"] == 1 and r2.output["count"] == 1 and r2.replayed and counter["n"] == 1
    r3 = reg.call("write_thing", {"k": "b"}, ctx)
    assert r3.output["count"] == 2


def test_compaction_timeout_and_errors(workspace, ctx):
    reg, _ = make_registry(workspace)
    r = reg.call("big", {}, ctx)
    assert r.ok and r.compacted and r.output["compacted"] and r.output["full_output_path"].startswith(".edgent/tool-results/")
    assert len(workspace.read_text(r.output["full_output_path"])) > 20_000
    r = reg.call("slow", {}, ctx)
    assert r.error.code == "timeout" and r.error.retryable and r.error.recovery["kind"] == "inspect_timeout_state"
    r = reg.call("fails", {}, ctx)
    assert r.error.code == "unavailable" and r.output == {"done": 3} and "install ffmpeg" in r.error.remediation
    r = reg.call("boom", {}, ctx)
    assert r.error.code == "execution_error" and "kaboom" in r.error.message
    r = reg.call("attach", {}, ctx)
    assert r.ok and r.attachments and r.attachments[0].type == "image" and "_attach" not in r.output
    d = r.to_dict()
    assert ToolResult.from_dict(d).attachments[0].path == r.attachments[0].path
    assert '"error"' in reg.call("boom", {}, ctx).as_text()


def test_parallel_and_schemas(workspace, ctx):
    reg, _ = make_registry(workspace)
    res = reg.call_parallel([("echo", {"text": "a"}, "c1"), ("echo", {"text": "b"}, "c2")], ctx)
    assert [r.output["text"] for r in res] == ["a", "b"] and res[0].call_id == "c1"
    schemas = reg.openai_schemas(allowlist={"echo"})
    assert len(schemas) == 1 and schemas[0]["function"]["name"] == "echo"
    man = reg.mcp_manifest(compact=True)
    assert man["tools"][0]["annotations"]["x-permission"] in ("read", "write", "delete")
    assert reg.schema_bytes(compact=True) <= reg.schema_bytes(compact=False)
    with pytest.raises(ValueError):
        reg.register(ToolSpec("echo", "dup", obj({}), lambda c, a: None))


def test_compact_schema():
    schema = obj({"a": {"type": "string", "description": "the a", "examples": ["x"]}, "b": obj({"c": {"type": "integer", "default": 1, "minimum": 0}})}, ["a"], description="root")
    c = compact_schema(schema)
    assert c["description"] == "root" and "description" not in c["properties"]["a"] and c["properties"]["b"]["properties"]["c"]["minimum"] == 0
    assert c["required"] == ["a"] and c["additionalProperties"] is False
