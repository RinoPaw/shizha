"""Agent package — intent routing, task dispatch, handlers."""

from .agent import Agent
from .router import IntentRouter
from .models import (
    TaskType,
    TaskConfig,
    AgentDecision,
    AgentResult,
    TaskResult,
    TASK_CONFIGS,
    _TASK_CONFIGS,
    task_type_from_str,
    task_type_label,
)
from .planner import (
    agent_planner_extra_options,
    build_agent_planner_messages,
    call_agent_planner_model,
    clamp_float,
    decision_from_planner_payload,
    extract_json_object,
)
from .formatting import (
    format_context_item_for_llm,
    items_to_llm_context,
    items_to_title_context,
    context_title_keywords,
)

__all__ = [
    "Agent",
    "AgentDecision",
    "AgentResult",
    "IntentRouter",
    "TaskConfig",
    "TaskResult",
    "TaskType",
    "TASK_CONFIGS",
    "_TASK_CONFIGS",
    "agent_planner_extra_options",
    "build_agent_planner_messages",
    "call_agent_planner_model",
    "clamp_float",
    "decision_from_planner_payload",
    "extract_json_object",
    "format_context_item_for_llm",
    "items_to_llm_context",
    "items_to_title_context",
    "context_title_keywords",
    "task_type_from_str",
    "task_type_label",
]
