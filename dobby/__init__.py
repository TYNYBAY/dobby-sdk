"""Dobby: a provider-neutral agentic LLM SDK for OpenAI and Gemini.

Exposes the :class:`AgentExecutor` agentic loop plus the opt-in context-compaction
surface (:class:`ContextPolicy`, :class:`ContextEditEvent`, and the agent-invoked
:class:`CompactContextTool`).
"""

from .context import ContextPolicy as ContextPolicy
from .executor import AgentExecutor as AgentExecutor
from .tools import CompactContextTool as CompactContextTool
from .types import ContextEditEvent as ContextEditEvent
