"""Tool registry and general-purpose toolset."""
from .registry import ToolSpec, ToolRegistry, ToolContext, ToolFailure, IdempotencyStore, PERMISSION_CLASSES  # noqa: F401
from .builtin import build_default_registry  # noqa: F401
from .general import register_general_tools  # noqa: F401
from .shell import BackgroundTaskManager, bash_tool, task_output_tool, task_status_tool, task_stop_tool  # noqa: F401
from .web import html_to_text, web_fetch, web_search  # noqa: F401
from .mcp_server import MCPServer  # noqa: F401
from .mcp_client import MCPClient  # noqa: F401
