"""Client-side context compaction for OpenAI and Gemini.

Exposes :class:`ContextPolicy` (config), :func:`edit_context` (deterministic
trim), and :func:`summarize_context` (LLM-backed write-back summarize).
"""

from .edit import edit_context as edit_context
from .policy import ContextPolicy as ContextPolicy
from .summarize import (
    SUMMARIZE_PROMPT as SUMMARIZE_PROMPT,
    summarize_context as summarize_context,
)
