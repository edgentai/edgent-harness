"""Five sub-agent roles (§13.1) and their code-enforced tool allowlists (§8.5)."""
from __future__ import annotations

READ_ONLY = {"read_file", "list_files", "search_files", "fetch_artifact_view", "list_artifacts", "search_memory", "video_probe", "transcript_between", "count_matches"}
MEDIA = {"crop_zoom", "ocr_region", "describe_region", "compare_images", "video_scenes", "frame_at", "frames_between", "transcribe_range", "diarise", "tag_audio_events"}
COMPUTE = {"run_python", "run_tests", "lint_code", "calculate"}
WRITE = {"write_file"}
META = {"spawn_subagent", "ask_user"}

ROLE_TOOLS: dict[str, set[str]] = {
    "intake": READ_ONLY | MEDIA | {"calculate", "count_matches"},
    "planner": READ_ONLY,
    "executor": READ_ONLY | MEDIA | COMPUTE | WRITE | META,
    "verifier": READ_ONLY | MEDIA | COMPUTE,  # read-only plus re-run tools; never write or spawn
    "synthesiser": READ_ONLY,
}


def allowlist_for(role: str, sub_agents_allowed: bool, interactive: bool) -> set[str]:
    base = set(ROLE_TOOLS.get(role, READ_ONLY))
    if not sub_agents_allowed:
        base.discard("spawn_subagent")
    if not interactive:
        base.discard("ask_user")
    return base
