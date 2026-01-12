"""LLM provider initialization and factory module.

This module provides a factory function to create LLM provider instances
based on application settings. Currently supports OpenAI and Azure OpenAI.
"""

from .executor import AgentExecutor as AgentExecutor
