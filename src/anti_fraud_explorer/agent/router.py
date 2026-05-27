"""Intent router — calls the planner model to decide user intent."""

from ..domain.dataset import KnowledgeBase
from .models import AgentDecision
from .planner import call_agent_planner_model


class IntentRouter:
    """Plan user input with the model planner."""

    def decide(
        self,
        query: str,
        kb: KnowledgeBase,
        category: str = "",
        context: dict | None = None,
    ) -> AgentDecision:
        return call_agent_planner_model(query, kb, category, context)

    def plan(
        self,
        query: str,
        kb: KnowledgeBase,
        category: str = "",
        context: dict | None = None,
    ) -> AgentDecision:
        return self.decide(query, kb, category, context)

    def needs_retrieval(
        self,
        query: str,
        kb: KnowledgeBase,
        category: str = "",
        context: dict | None = None,
    ) -> bool:
        return self.decide(query, kb, category, context).needs_retrieval
