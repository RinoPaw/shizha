"""Shared contracts and task configurations for the anti-fraud agent."""

import enum
from dataclasses import dataclass, field
from typing import Any


class TaskType(enum.Enum):
    CHITCHAT = "chitchat"
    FACT_QA = "fact_qa"
    BROWSE_QUERY = "browse_query"
    COMPARISON = "comparison"
    RECOMMENDATION = "recommendation"
    LECTURE_PLAN = "lecture_plan"
    STUDY_TASK = "study_task"
    CONTENT_TRANSFORM = "content_transform"

    # backward-compat enum names
    FACTUAL_QA = FACT_QA
    CURRICULUM_DESIGN = STUDY_TASK
    CREATIVE_BRIEF = CONTENT_TRANSFORM
    DATA_EXPLORE = BROWSE_QUERY
    MULTI_FILTER = BROWSE_QUERY


@dataclass(frozen=True)
class AgentDecision:
    """A small, explicit plan for what the agent should do next."""

    task_type: TaskType
    confidence: float
    needs_retrieval: bool
    needs_llm: bool
    reason: str
    direct_answer: str = ""
    mode: str = "local"
    warnings: list[str] = field(default_factory=list)
    planner: str = "offline"
    search_queries: list[str] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        return {
            "task_type": self.task_type.value,
            "confidence": self.confidence,
            "needs_retrieval": self.needs_retrieval,
            "needs_llm": self.needs_llm,
            "reason": self.reason,
            "mode": self.mode,
            "warnings": list(self.warnings),
            "planner": self.planner,
            "search_queries": list(self.search_queries),
        }


@dataclass(frozen=True)
class TaskConfig:
    task_type: TaskType
    retrieval_limit: int = 5
    require_diversity: bool = False
    context_schema: str = "fact_sheet"
    handler_name: str | None = None
    generate_detail: str = "正在请模型生成最终文字"


@dataclass
class AgentResult:
    task_type: TaskType
    answer: str = ""
    items: list[dict[str, Any]] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    selection_reason: str = ""
    mode: str = "local"
    confidence: float = 0.0
    warnings: list[str] = field(default_factory=list)
    total_count: int = 0
    decision: dict[str, Any] = field(default_factory=dict)
    # backward-compat
    speech: str = ""


# backward-compat alias
TaskResult = AgentResult

_TASK_TYPE_LABELS = {
    TaskType.CHITCHAT: "对话回应",
    TaskType.FACT_QA: "事实问答",
    TaskType.BROWSE_QUERY: "资料筛选",
    TaskType.COMPARISON: "案例对比",
    TaskType.RECOMMENDATION: "案例推荐",
    TaskType.LECTURE_PLAN: "展示策划",
    TaskType.STUDY_TASK: "宣教任务",
    TaskType.CONTENT_TRANSFORM: "内容转化",
}


def task_type_label(task_type: TaskType) -> str:
    return _TASK_TYPE_LABELS.get(task_type, "事实问答")


def task_type_from_str(value: str) -> TaskType:
    try:
        return TaskType(value)
    except ValueError:
        return TaskType.FACT_QA


# ---------------------------------------------------------------------------
# Task execution settings
# ---------------------------------------------------------------------------

TASK_CONFIGS: dict[TaskType, TaskConfig] = {
    TaskType.CHITCHAT: TaskConfig(
        task_type=TaskType.CHITCHAT,
        retrieval_limit=0,
        context_schema="none",
        generate_detail="正在整理上下文，准备简短回应",
    ),
    TaskType.FACT_QA: TaskConfig(
        task_type=TaskType.FACT_QA,
        retrieval_limit=5,
        context_schema="fact_sheet",
        generate_detail="正在组织证据并撰写依据式回答",
    ),
    TaskType.BROWSE_QUERY: TaskConfig(
        task_type=TaskType.BROWSE_QUERY,
        retrieval_limit=30,
        context_schema="fact_sheet",
        handler_name="_handle_browse",
        generate_detail="正在汇总条目，整理成可浏览清单",
    ),
    TaskType.COMPARISON: TaskConfig(
        task_type=TaskType.COMPARISON,
        retrieval_limit=8,
        require_diversity=True,
        context_schema="comparison_table",
        handler_name="_handle_comparison",
        generate_detail="正在提取差异点并整理对比表格",
    ),
    TaskType.RECOMMENDATION: TaskConfig(
        task_type=TaskType.RECOMMENDATION,
        retrieval_limit=10,
        require_diversity=True,
        context_schema="recommendation_cards",
        handler_name="_handle_recommend",
        generate_detail="正在按场景筛选、排序并写推荐理由",
    ),
    TaskType.LECTURE_PLAN: TaskConfig(
        task_type=TaskType.LECTURE_PLAN,
        retrieval_limit=5,
        context_schema="exhibition_brief",
        handler_name="_handle_lecture",  # 已重命名
        generate_detail="正在选取合适案例并编排宣传流程",
    ),
    TaskType.STUDY_TASK: TaskConfig(
        task_type=TaskType.STUDY_TASK,
        retrieval_limit=5,
        context_schema="curriculum_brief",
        handler_name="_handle_study_task",
        generate_detail="正在把案例资料转成课堂任务",
    ),
    TaskType.CONTENT_TRANSFORM: TaskConfig(
        task_type=TaskType.CONTENT_TRANSFORM,
        retrieval_limit=5,
        context_schema="creative_brief",
        handler_name="_handle_content_transform",
        generate_detail="正在围绕目标案例改写成指定文体",
    ),
}

# Backward-compatible name used by existing tests/imports.
_TASK_CONFIGS = TASK_CONFIGS
